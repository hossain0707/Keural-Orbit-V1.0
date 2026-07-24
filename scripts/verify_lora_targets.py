#!/usr/bin/env python
"""Keural Orbit — Gemma 4 LoRA-targeting safety gate (CPU/meta-device ONLY).

The raison d'être of this pipeline: MoE LoRA on Gemma 4 fails SILENTLY when
misconfigured. The 128 routed experts are fused nn.Parameter tensors
(experts.gate_up_proj / experts.down_proj), reachable only via peft
`target_parameters` — a plain target_modules list never touches them, so training
runs, loss drops, checkpoints save, and most of the model's FFN capacity is never
adapted. Conversely, adapting the router (router.proj), the vision tower
(Gemma4ClippableLinear), or the tied embeddings corrupts the model. This gate
hard-verifies the RESOLVED adapter placement instead of trusting name suffixes
(VERIFIED-FACTS-ORBIT.md §2/§3).

Two modes, both CPU/meta-device only — never loads weights, never touches GPUs:

  --pre --train-config configs/train_orbit_pilot.yaml
      Reads the LoRA fields from the YAML (never hardcoded), instantiates the REAL
      30-layer Gemma4ForConditionalGeneration on torch.device('meta') (meta tensors
      have no storage → free), attaches peft LoRA exactly as configured, and asserts
      the resulting trainable-parameter set.

  --checkpoint <dir>
      Re-runs the same structural assertions from the checkpoint's own
      adapter_config.json (catches ms-swift translating the YAML into something
      else), then opens adapter_model.safetensors (key metadata only, no tensor
      loads) and asserts the actually-saved key set.

Exit 0 only if ALL checks pass. JSON report -> logs/verify/<stamp>.json (+ --json-out).
"""
from __future__ import annotations

import os

# Belt and braces: never see a GPU even if a caller leaks CUDA_VISIBLE_DEVICES
# (the Nova gateway / another job may own them). Must precede any torch import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

STAGE = "verify"
DEFAULT_TRAIN_CONFIG = "configs/train_orbit_pilot.yaml"


# --- lora-field extraction ---------------------------------------------------


def lora_fields_from_yaml(cfg: dict) -> dict:
    """LoRA targeting fields from the train YAML — never hardcoded, never defaulted.

    ms-swift defaults are hazards here (target_modules->all-linear crashes on the
    vision tower; lora_dropout 0.05 crashes peft ParamWrapper), so require explicit.
    """
    sa = cfg.get("swift_args")
    if not isinstance(sa, dict):
        raise common.PipelineError("train config has no swift_args mapping")
    if not (sa.get("target_regex") or sa.get("target_modules")):
        raise common.PipelineError(
            "swift_args must set target_regex (preferred for Gemma 4 — scopes to the "
            "language model) or target_modules; the gate refuses implicit all-linear "
            "targeting (crashes on the vision tower's Gemma4ClippableLinear)."
        )
    for k in ("lora_rank", "lora_alpha", "lora_dropout"):
        if k not in sa:
            raise common.PipelineError(f"swift_args missing required LoRA field {k!r}")
    return {
        "model": sa["model"],
        "target_modules": sa.get("target_regex") or list(sa["target_modules"]),
        "target_parameters": list(sa.get("target_parameters") or []),
        "lora_rank": int(sa["lora_rank"]),
        "lora_alpha": int(sa["lora_alpha"]),
        "lora_dropout": float(sa["lora_dropout"]),
    }


def lora_fields_from_adapter_config(path: Path, model_id: str) -> dict:
    """What ms-swift ACTUALLY handed peft (adapter_config.json) — catches drift."""
    if not path.is_file():
        raise common.PipelineError(
            f"{path} not found — not a LoRA checkpoint dir? Expected the peft "
            "adapter_config.json next to adapter_model.safetensors."
        )
    ac = json.loads(path.read_text(encoding="utf-8"))
    for k in ("r", "lora_alpha", "target_modules"):
        if k not in ac:
            raise common.PipelineError(f"{path} missing required peft field {k!r}")
    tm = ac["target_modules"]
    return {
        "model": model_id,
        "target_modules": tm if isinstance(tm, str) else list(tm),
        "target_parameters": list(ac.get("target_parameters") or []),
        "lora_rank": int(ac["r"]),
        "lora_alpha": int(ac["lora_alpha"]),
        "lora_dropout": float(ac.get("lora_dropout", 0.0)),
    }


# --- meta-device attach + checks ---------------------------------------------


def topology(model_id: str, logger):
    """AutoConfig -> (config, n_layers, full_attention_layers). Local cache first."""
    from transformers import AutoConfig

    try:
        conf = AutoConfig.from_pretrained(model_id, local_files_only=True)
        logger.info("model config from local HF cache (%s)", model_id)
    except Exception as e:  # noqa: BLE001 — cache miss falls through to the Hub
        logger.warning("local cache miss (%s); fetching config from Hub", e)
        conf = AutoConfig.from_pretrained(model_id)
    tc = conf.text_config
    n_layers = tc.num_hidden_layers
    layer_types = getattr(tc, "layer_types", []) or []
    full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    return conf, n_layers, full_attn


def attach_on_meta(conf, lora: dict, logger) -> list[str]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Gemma4ForConditionalGeneration

    logger.info("instantiating Gemma4ForConditionalGeneration on meta device...")
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration._from_config(conf)
    lc = LoraConfig(
        r=lora["lora_rank"], lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"], bias="none",
        target_modules=lora["target_modules"],
        target_parameters=lora["target_parameters"] or None,
    )
    pm = get_peft_model(model, lc)
    return [n for n, p in pm.named_parameters() if p.requires_grad]


def run_checks(names: list[str], lora: dict, n_layers: int, kind: str) -> dict:
    """Evaluate ALL assertions (no early exit) over a set of adapter names.

    `names` are live peft trainable names (kind='structural') or saved
    adapter_model.safetensors keys (kind='weights'); both share stable substrings.
    """
    checks: list[dict] = []

    def check(cid: str, passed: bool, detail: str) -> None:
        checks.append({"id": cid, "passed": bool(passed), "detail": detail})

    def layers_with(pat: str) -> set[int]:
        return {int(m.group(1)) for n in names
                if (m := re.search(rf"layers\.(\d+)\.{pat}", n))}

    all_layers = set(range(n_layers))
    n_tp = len(lora["target_parameters"])

    # dropout must be 0 with target_parameters (peft ParamWrapper crashes otherwise)
    check("dropout_zero_with_target_parameters",
          not (n_tp and lora["lora_dropout"] != 0.0),
          f"dropout={lora['lora_dropout']}, target_parameters={lora['target_parameters']}")

    router = [n for n in names if "router" in n]
    check("router_frozen", not router, f"{len(router)} trainable" if router else "clean")

    vision = [n for n in names if "vision_tower" in n or "embed_vision" in n]
    check("vision_frozen", not vision, f"{len(vision)} trainable" if vision else "clean")

    emb = [n for n in names if "embed_tokens" in n or "lm_head" in n]
    check("embed_head_frozen", not emb, f"{len(emb)} trainable" if emb else "clean")

    exp = layers_with(r"experts")
    check("experts_all_layers", exp == all_layers,
          f"experts on {len(exp)}/{n_layers} layers"
          + ("" if exp == all_layers else f" MISSING {sorted(all_layers - exp)}"))

    attn = layers_with(r"self_attn")
    check("attention_all_layers", attn == all_layers,
          f"attn on {len(attn)}/{n_layers} layers")

    mlp = layers_with(r"mlp\.(gate|up|down)_proj")
    check("dense_mlp_all_layers", mlp == all_layers,
          f"dense mlp on {len(mlp)}/{n_layers} layers")

    def expected(n: str) -> bool:
        return ("language_model.layers" in n and
                ("self_attn" in n or ".mlp." in n or "experts" in n))
    stray = [n for n in names if not expected(n)]
    check("no_unexpected_trainables", not stray, f"{stray[:2]}" if stray else "clean")

    return {"kind": kind, "n_names": len(names), "checks": checks,
            "passed": all(c["passed"] for c in checks)}


def saved_adapter_keys(ckpt: Path) -> list[str]:
    st = ckpt / "adapter_model.safetensors"
    if not st.is_file():
        raise common.PipelineError(f"{st} not found — not a saved LoRA checkpoint.")
    from safetensors import safe_open

    with safe_open(str(st), framework="pt", device="cpu") as f:
        return list(f.keys())


# --- reporting ---------------------------------------------------------------


def finalize(report: dict, cfg: dict, logger, json_out: str | None) -> int:
    stamp = common.utc_stamp()
    report["timestamp_utc"] = stamp
    logs_dir = common.resolve_path(cfg, cfg["pipeline"]["paths"]["logs_dir"]) / "verify"
    common.write_json_atomic(logs_dir / f"{stamp}.json", report)
    if json_out:
        common.write_json_atomic(common.resolve_path(cfg, json_out), report)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    if not report["passed"]:
        failed = [c["id"] for s in report["sections"] for c in s["checks"] if not c["passed"]]
        logger.error("VERIFICATION FAILED — %s. Do NOT train/merge on this.", failed)
        print(f"\nFAILED CHECKS: {failed}", file=sys.stderr)
        return 1
    logger.info("all checks passed")
    print("\nALL CHECKS PASSED ✅  (router + vision + embeddings frozen; experts/attn/mlp adapted)")
    return 0


# --- modes -------------------------------------------------------------------


def mode_pre(train_config: str, json_out: str | None) -> int:
    cfg = common.load_config(train_config)
    logger = common.setup_logging(STAGE, cfg)
    common.stage_banner(logger, "verify_lora_targets --pre (Gemma 4)", train_config)
    common.guard_environment(logger)

    lora = lora_fields_from_yaml(cfg)
    conf, n_layers, full_attn = topology(lora["model"], logger)

    # Attach with dropout forced to 0 so the structural checks still all run even
    # when dropout is misconfigured (the dropout check itself still records FAIL).
    attach = dict(lora, lora_dropout=0.0)
    names = attach_on_meta(conf, attach, logger)
    section = run_checks(names, lora, n_layers, "structural (meta-device peft attach)")

    report = {"mode": "pre", "train_config": str(train_config), "lora": lora,
              "model": lora["model"], "n_layers": n_layers,
              "full_attention_layers": full_attn, "sections": [section],
              "passed": section["passed"]}
    return finalize(report, cfg, logger, json_out)


def mode_checkpoint(checkpoint: str, train_config: str, json_out: str | None) -> int:
    cfg = common.load_config(train_config)
    logger = common.setup_logging(STAGE, cfg)
    common.stage_banner(logger, "verify_lora_targets --checkpoint (Gemma 4)", checkpoint)
    common.guard_environment(logger)

    ckpt = common.resolve_path(cfg, checkpoint)
    if not ckpt.is_dir():
        raise common.PipelineError(f"checkpoint dir not found: {ckpt}")
    model_id = cfg["pipeline"]["model"]["id"]
    lora = lora_fields_from_adapter_config(ckpt / "adapter_config.json", model_id)
    conf, n_layers, full_attn = topology(model_id, logger)

    attach = dict(lora, lora_dropout=0.0)
    names = attach_on_meta(conf, attach, logger)
    structural = run_checks(names, lora, n_layers,
                            "structural (re-attach from adapter_config.json)")
    weights = run_checks(saved_adapter_keys(ckpt), lora, n_layers,
                         "weights (adapter_model.safetensors key set)")

    report = {"mode": "checkpoint", "checkpoint": str(ckpt), "lora": lora,
              "model": model_id, "n_layers": n_layers,
              "full_attention_layers": full_attn, "sections": [structural, weights],
              "passed": structural["passed"] and weights["passed"]}
    return finalize(report, cfg, logger, json_out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 4 LoRA verification gate "
                                 "(CPU/meta-device only; never touches GPUs).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pre", action="store_true",
                      help="verify the LoRA targeting declared in --train-config")
    mode.add_argument("--checkpoint", metavar="DIR",
                      help="verify a saved ms-swift/peft LoRA checkpoint dir")
    ap.add_argument("--train-config", default=DEFAULT_TRAIN_CONFIG)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    try:
        if args.pre:
            return mode_pre(args.train_config, args.json_out)
        return mode_checkpoint(args.checkpoint, args.train_config, args.json_out)
    except common.PipelineError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
