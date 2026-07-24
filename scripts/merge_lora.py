#!/usr/bin/env python
"""Keural Orbit — CPU-only LoRA merge for Gemma 4 (google/gemma-4-26B-A4B-it).

WHY this is not a three-line merge_and_unload snippet:

* MoE LoRA fails SILENTLY when misconfigured — the 128 routed experts are fused
  ``nn.Parameter`` tensors (``experts.gate_up_proj`` / ``experts.down_proj``)
  handled by peft's ``lora.ParamWrapper``. The merge verifier PROVES targeted
  tensors actually CHANGED vs base and untargeted tensors (router, vision tower,
  embeddings, norms) are byte-identical — catching both a no-op merge and a merge
  that corrupted frozen weights.

* Gemma 4 differs from the Keural Nova (Qwen) pipeline in two load-bearing ways
  (VERIFIED-FACTS-ORBIT §4):
  - **No MTP block** — nothing to re-inject (Qwen needed mtp.* re-injection).
  - **Tied word embeddings** (``tie_word_embeddings: true``) — ``save_pretrained``
    writes NO separate ``lm_head.weight``; the base repo likewise omits it, so the
    tensor-set-equality check holds. ``save_original_format=False`` keeps the fused
    expert tensor names (verified: a tiny fixture round-trips fused names + omits
    lm_head).

peft 0.19.1 ParamWrapper contract is re-audited dynamically (paramwrapper_findings)
so a peft re-pin fails LOUDLY instead of producing a half-merged model.

CPU-only by construction: ``CUDA_VISIBLE_DEVICES`` is cleared before torch import
(the Nova gateway / another job may own the GPUs).
"""
from __future__ import annotations

import os

# MUST precede any torch/transformers import — the GPUs are not ours to touch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import PipelineError  # noqa: E402

DEFAULT_CONFIG = "configs/publish.yaml"

# Files shipped alongside the merged weights (Gemma 4 is multimodal — the vision
# processor files must ship). config.json / generation_config.json written by
# save_pretrained are never clobbered.
AUX_PATTERNS = [
    "tokenizer*",
    "*.model",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
]


# ------------------------------------------------------------------ peft audit


def paramwrapper_findings() -> str:
    """Dynamically re-verify the installed peft ParamWrapper merge contract."""
    import inspect

    import peft
    from peft.tuners.lora.layer import ParamWrapper

    problems = []
    for attr in ("merge", "get_delta_weight", "unload_and_optionally_merge_module"):
        if not hasattr(ParamWrapper, attr):
            problems.append(f"ParamWrapper.{attr} missing")
    src = inspect.getsource(ParamWrapper.unload_and_optionally_merge_module) if not problems else ""
    if src and "while isinstance(base_layer, ParamWrapper)" not in src:
        problems.append(
            "ParamWrapper.unload_and_optionally_merge_module no longer merges nested "
            "wrappers recursively — double-wrapped expert adapters would be dropped"
        )
    if problems:
        raise PipelineError(
            f"peft {peft.__version__} ParamWrapper contract changed: {problems}. "
            "Re-audit peft/tuners/lora/layer.py before merging (a wrong merge is the "
            "silent failure this pipeline exists to prevent)."
        )
    return (f"peft {peft.__version__}: ParamWrapper.merge applies per-expert einsum "
            "delta in-place (transposed-layout aware); nested double-wrap merged "
            "recursively — merge_and_unload verified safe for fused experts.")


# ------------------------------------------------------- safetensors utilities


def read_safetensors_header(path: Path) -> dict[str, dict]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len).decode("utf-8"))
    header.pop("__metadata__", None)
    out = {}
    for name, meta in header.items():
        out[name] = {"dtype": meta["dtype"], "shape": tuple(meta["shape"]),
                     "offsets": tuple(meta["data_offsets"]), "file": path,
                     "data_start": 8 + header_len}
    return out


def model_tensor_map(model_dir: Path) -> dict[str, dict]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = common.read_json(index_path)["weight_map"]
        tmap: dict[str, dict] = {}
        for shard in sorted(set(weight_map.values())):
            shard_path = model_dir / shard
            if not shard_path.exists():
                raise PipelineError(f"Shard in index but missing on disk: {shard_path}")
            tmap.update(read_safetensors_header(shard_path))
        missing = set(weight_map) - set(tmap)
        if missing:
            raise PipelineError(f"{len(missing)} tensors in index but absent from shards "
                                f"of {model_dir}: {sorted(missing)[:5]}...")
        return tmap
    single = model_dir / "model.safetensors"
    if single.exists():
        return read_safetensors_header(single)
    raise PipelineError(f"No model.safetensors[.index.json] under {model_dir}.")


def read_tensor_bytes(meta: dict) -> bytes:
    with open(meta["file"], "rb") as f:
        f.seek(meta["data_start"] + meta["offsets"][0])
        return f.read(meta["offsets"][1] - meta["offsets"][0])


# ------------------------------------------------------------- target analysis


def load_adapter_targets(checkpoint: Path) -> tuple[list | str, list]:
    cfg_path = checkpoint / "adapter_config.json"
    if not cfg_path.exists():
        raise PipelineError(
            f"{cfg_path} not found — --checkpoint must point at a LoRA checkpoint dir "
            "(e.g. outputs/full/<run>/checkpoint-XXXX) containing adapter_config.json.")
    acfg = common.read_json(cfg_path)
    return acfg.get("target_modules") or [], acfg.get("target_parameters") or []


def make_target_predicate(target_modules, target_parameters):
    """Predicate: is a base tensor expected to CHANGE on merge? (regex-string aware).

    Gemma 4 uses a target_regex string scoped to the language model. Conservative:
    misclassifying an adapted tensor as untargeted makes the byte check FAIL LOUDLY.
    """
    params = set(target_parameters or [])
    if isinstance(target_modules, str):
        pattern = re.compile(target_modules)

        def module_hit(mod_path: str) -> bool:
            return bool(pattern.fullmatch(mod_path))
    else:
        modules = set(target_modules or [])

        def module_hit(mod_path: str) -> bool:
            return mod_path.rsplit(".", 1)[-1] in modules

    def is_targeted(name: str) -> bool:
        last = name.rsplit(".", 1)[-1]
        if last in params and not name.endswith(".weight"):
            return True  # fused expert nn.Parameter (experts.gate_up_proj/down_proj)
        if name.endswith(".weight") and module_hit(name[: -len(".weight")]):
            return True
        return False

    return is_targeted


# ------------------------------------------------------------------ aux + config


def copy_aux_files(base_dir: Path, out_dir: Path, logger) -> list[str]:
    copied = []
    for pattern in AUX_PATTERNS:
        for src in sorted(base_dir.glob(pattern)):
            if not src.is_file():
                continue
            dst = out_dir / src.name
            if dst.exists():
                logger.info("aux: %s already written by save_pretrained — kept", src.name)
                continue
            shutil.copy2(src, dst)
            copied.append(src.name)
    logger.info("aux files copied from base snapshot: %s", copied or "none")
    return copied


def diff_config_json(base_dir: Path, out_dir: Path, logger) -> dict:
    def flatten(d, prefix=""):
        out = {}
        for k, v in sorted(d.items()):
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flatten(v, key + "."))
            else:
                out[key] = v
        return out

    base = flatten(common.read_json(base_dir / "config.json"))
    merged = flatten(common.read_json(out_dir / "config.json"))
    diff = {}
    for key in sorted(set(base) | set(merged)):
        b, m = base.get(key, "<absent>"), merged.get(key, "<absent>")
        if b != m:
            diff[key] = {"base": b, "merged": m}
            logger.info("config.json diff: %s: base=%r merged=%r", key, b, m)
    return diff


# ---------------------------------------------------------------- verification


def verify_merged(base_dir: Path, out_dir: Path, target_modules, target_parameters,
                  logger, seed: int = 42, n_untargeted: int = 5, n_targeted: int = 3) -> dict:
    """Post-merge verification — ALWAYS runs; raises on any failure.

    (1) tensor-name-set + shape/dtype equality vs base (tied lm_head absent in both);
    (2) N random UNTARGETED tensors byte-identical (router + one vision tensor forced);
    (3) N TARGETED tensors must NOT be identical (a no-op merge is the silent failure);
    (4) tokenizer/processor aux files present.
    """
    rng = random.Random(seed)
    base_map = model_tensor_map(base_dir)
    out_map = model_tensor_map(out_dir)
    report: dict = {}

    missing = sorted(set(base_map) - set(out_map))
    extra = sorted(set(out_map) - set(base_map))
    if missing or extra:
        raise PipelineError(
            f"Tensor set mismatch vs base: {len(missing)} missing (e.g. {missing[:5]}), "
            f"{len(extra)} extra (e.g. {extra[:5]}). Merged model would silently differ "
            "from the base architecture.")
    bad = [n for n in base_map if base_map[n]["shape"] != out_map[n]["shape"]
           or base_map[n]["dtype"] != out_map[n]["dtype"]]
    if bad:
        details = {n: (base_map[n]["shape"], base_map[n]["dtype"],
                       out_map[n]["shape"], out_map[n]["dtype"]) for n in bad[:5]}
        raise PipelineError(f"{len(bad)} tensors changed shape/dtype vs base: {details}")
    report["tensor_set"] = {"count": len(base_map), "match": True}
    logger.info("check 1 OK: %d tensor names, shapes and dtypes identical to base", len(base_map))

    is_targeted = make_target_predicate(target_modules, target_parameters)
    targeted = sorted(n for n in base_map if is_targeted(n))
    untargeted = sorted(n for n in base_map if not is_targeted(n))
    if len(targeted) < n_targeted:
        raise PipelineError(
            f"Only {len(targeted)} tensors resolve as LoRA-targeted "
            f"(target_modules={target_modules!r}, target_parameters={target_parameters!r}) "
            "— the adapter targeted (almost) nothing. Silent MoE-LoRA failure; refusing "
            "to publish.")

    forced = [n for n in untargeted if n.endswith("router.proj.weight")][:1]
    forced += [n for n in untargeted if ".vision_tower." in n][:1]
    pool = [n for n in untargeted if n not in forced]
    sample_untargeted = forced + rng.sample(pool, min(n_untargeted, len(pool)))
    for n in sample_untargeted:
        if read_tensor_bytes(base_map[n]) != read_tensor_bytes(out_map[n]):
            raise PipelineError(
                f"UNTARGETED tensor {n!r} differs from base after merge — the merge "
                "touched weights it must not (router/vision/embeddings/frozen params).")
    report["untargeted_byte_identical"] = sample_untargeted
    logger.info("check 2 OK: %d untargeted tensors byte-identical (incl. %s)",
                len(sample_untargeted), forced)

    sample_targeted = rng.sample(targeted, n_targeted)
    unchanged = [n for n in sample_targeted
                 if read_tensor_bytes(base_map[n]) == read_tensor_bytes(out_map[n])]
    if unchanged:
        raise PipelineError(
            f"TARGETED tensors identical to base after merge: {unchanged}. The LoRA "
            "delta was not applied (zero adapter? wrong checkpoint? peft regression). "
            "A no-op merge is the silent failure this pipeline exists to catch.")
    report["targeted_changed"] = sample_targeted
    logger.info("check 3 OK: %d targeted tensors differ from base: %s", n_targeted, sample_targeted)

    present, absent = [], []
    for pattern in AUX_PATTERNS:
        for name in [p.name for p in base_dir.glob(pattern) if p.is_file()]:
            (present if (out_dir / name).exists() else absent).append(name)
    if absent:
        raise PipelineError(
            f"Aux files in base snapshot but missing from merged dir: {absent}. The "
            "multimodal processor/tokenizer files must ship.")
    if not (out_dir / "config.json").exists():
        raise PipelineError(f"config.json missing from {out_dir}")
    report["aux_files_present"] = sorted(set(present))
    logger.info("check 4 OK: aux files present: %s", sorted(set(present)))
    return report


# ------------------------------------------------------------------ merge core


def ram_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    raise PipelineError("MemAvailable not found in /proc/meminfo")


def run_merge(base_dir: Path, checkpoint: Path, out_dir: Path, logger,
              dtype_name: str = "bfloat16", max_shard_size: str = "4GB",
              copy_aux: bool = True, allow_no_target_parameters: bool = False) -> dict:
    """Load base -> attach adapter -> merge_and_unload -> save -> copy aux -> verify.

    Pure function of explicit paths so the tiny-fixture self-test exercises the
    EXACT production code path. No MTP re-injection (Gemma 4 has none)."""
    import torch

    if torch.cuda.is_available():  # CVD="" makes this False without touching GPUs
        raise PipelineError("CUDA visible despite CUDA_VISIBLE_DEVICES='' — refusing "
                            "(GPUs belong to another job / the Nova gateway).")

    target_modules, target_parameters = load_adapter_targets(checkpoint)
    if not target_parameters and not allow_no_target_parameters:
        raise PipelineError(
            f"adapter_config.json at {checkpoint} has empty target_parameters — the "
            "routed experts were NOT adapted (the classic silent MoE failure). If "
            "genuinely intended, re-run with --allow-no-target-parameters.")

    from transformers import Gemma4ForConditionalGeneration

    timings: dict[str, float] = {}
    t0 = time.time()
    logger.info("loading base model from %s (dtype=%s, cpu)...", base_dir, dtype_name)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        str(base_dir), dtype=getattr(torch, dtype_name), low_cpu_mem_usage=True).eval()
    timings["load_base_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    from peft import PeftModel

    logger.info("attaching adapter from %s ...", checkpoint)
    peft_model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    logger.info("merging (safe_merge=True: NaN detection on every merged tensor)...")
    merged = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    timings["merge_s"] = round(time.time() - t0, 1)

    leftover = [n for n, _ in merged.named_modules() if "lora_" in n or "base_layer" in n]
    if leftover:
        raise PipelineError(f"peft wrappers survived merge_and_unload: {leftover[:5]}.")

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("saving merged model to %s (max_shard_size=%s)...", out_dir, max_shard_size)
    # save_original_format=False keeps the FUSED expert tensor names (experts.gate_up_proj)
    # matching the base repo layout — verified on a tiny fixture. tie_word_embeddings=True
    # => no separate lm_head.weight is written (base repo likewise omits it).
    merged.save_pretrained(str(out_dir), max_shard_size=max_shard_size,
                           safe_serialization=True, save_original_format=False)
    timings["save_s"] = round(time.time() - t0, 1)
    del merged, peft_model, model

    copied = copy_aux_files(base_dir, out_dir, logger) if copy_aux else []
    cfg_diff = diff_config_json(base_dir, out_dir, logger)

    t0 = time.time()
    checks = verify_merged(base_dir, out_dir, target_modules, target_parameters, logger)
    timings["verify_s"] = round(time.time() - t0, 1)

    return {"checks": checks, "aux_copied": copied, "config_json_diff": cfg_diff,
            "timings_sec": timings, "target_modules": target_modules,
            "target_parameters": target_parameters}


# -------------------------------------------------------------------- fixture


def build_tiny_fixture(work: Path, logger, seed: int = 42):
    """Shrunken Gemma4 (hidden 64, 2 layers [sliding, full], 8 experts, tiny vision)
    + randomized LoRA adapter with the REAL target spec from configs/train_orbit_full.yaml.
    Returns (base_dir, ckpt_dir, ref_logits, input_ids)."""
    import torch
    from transformers import Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig)

    torch.manual_seed(seed)
    txt = Gemma4TextConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        num_global_key_value_heads=2, head_dim=16, global_head_dim=16,
        enable_moe_block=True, num_experts=8, top_k_experts=2, vocab_size=256,
        max_position_embeddings=128, layer_types=["sliding_attention", "full_attention"],
        sliding_window=64, attention_k_eq_v=True, tie_word_embeddings=True)
    vis = Gemma4VisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                             intermediate_size=64, patch_size=16, image_size=32)
    cfg = Gemma4Config(text_config=txt, vision_config=vis, tie_word_embeddings=True)
    model = Gemma4ForConditionalGeneration(cfg).eval().float()

    base_dir = work / "tiny_base"
    base_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(base_dir), max_shard_size="500KB",
                          safe_serialization=True, save_original_format=False)
    for name in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
                 "preprocessor_config.json", "processor_config.json"):
        (base_dir / name).write_text('{"fixture": true}\n', encoding="utf-8")
    (base_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    logger.info("tiny Gemma4 base written at %s", base_dir)

    train_cfg = common.load_yaml("configs/train_orbit_full.yaml")["swift_args"]
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=train_cfg["target_regex"],
        target_parameters=list(train_cfg["target_parameters"]))
    peft_model = get_peft_model(model, lora_cfg)

    n_rand = 0
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:  # lora_B is zero-init => randomize
                torch.nn.init.normal_(p, std=0.05)
                n_rand += 1
    if n_rand == 0:
        raise PipelineError("fixture: no lora_A/lora_B parameters after attach")
    logger.info("fixture: randomized %d adapter tensors", n_rand)

    wrapped = [n for n, m in peft_model.named_modules() if type(m).__name__ == "ParamWrapper"]
    if len(wrapped) < 4:  # 2 layers x 2 fused params
        raise PipelineError(f"fixture: expected >=4 ParamWrapper modules, got {len(wrapped)}")

    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, 250, (2, 16))
    with torch.no_grad():
        ref_logits = peft_model(input_ids=input_ids).logits.clone()

    ckpt_dir = work / "tiny_ckpt"
    peft_model.save_pretrained(str(ckpt_dir))
    if not (ckpt_dir / "adapter_config.json").exists():
        sub = ckpt_dir / "default"
        if (sub / "adapter_config.json").exists():
            ckpt_dir = sub
        else:
            raise PipelineError(f"fixture: adapter_config.json not found under {ckpt_dir}")
    return base_dir, ckpt_dir, ref_logits, input_ids


def self_test(logger) -> None:
    """End-to-end CPU validation of the production merge path on a tiny Gemma4 fixture.

    Proves: untargeted tensors stay byte-identical, targeted tensors change, and the
    LoRA math — forward outputs of (base+adapter) vs (merged) agree to atol 1e-4 fp32."""
    import tempfile

    import torch
    from transformers import Gemma4ForConditionalGeneration

    logger.info(paramwrapper_findings())
    with tempfile.TemporaryDirectory(prefix="orbit-merge-selftest-") as tmp:
        work = Path(tmp)
        base_dir, ckpt_dir, ref_logits, input_ids = build_tiny_fixture(work, logger)
        out_dir = work / "tiny_merged"

        run_merge(base_dir, ckpt_dir, out_dir, logger,
                  dtype_name="float32", max_shard_size="500KB")

        merged = Gemma4ForConditionalGeneration.from_pretrained(
            str(out_dir), dtype=torch.float32).eval()
        with torch.no_grad():
            merged_logits = merged(input_ids=input_ids).logits
        diff = (merged_logits - ref_logits).abs().max().item()
        logger.info("self-test: max|peft - merged| = %.3e (atol 1e-4, fp32)", diff)
        if diff > 1e-4:
            raise PipelineError(f"self-test FAILED: merged deviates from peft by {diff:.3e} > 1e-4")

        base = Gemma4ForConditionalGeneration.from_pretrained(
            str(base_dir), dtype=torch.float32).eval()
        with torch.no_grad():
            base_logits = base(input_ids=input_ids).logits
        base_diff = (merged_logits - base_logits).abs().max().item()
        logger.info("self-test: max|base - merged| = %.3e (must be >1e-3)", base_diff)
        if base_diff < 1e-3:
            raise PipelineError("self-test FAILED: merged == base — no-op merge")
        logger.info("SELF-TEST PASSED: Gemma4 merge verified end-to-end (equivalence "
                    "%.3e <= 1e-4, base deviation %.3e)", diff, base_diff)


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None,
                        help="LoRA checkpoint dir (overrides cfg merge.checkpoint)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="run the tiny-fixture CPU validation and exit")
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--min-free-ram-gb", type=float, default=120.0)
    parser.add_argument("--allow-no-target-parameters", action="store_true")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("merge", cfg)
    common.stage_banner(logger, "merge_lora (Gemma 4)", args.config)
    common.guard_environment(logger)

    try:
        if args.self_test:
            self_test(logger)
            return

        mcfg = cfg.get("merge") or {}
        checkpoint = args.checkpoint or mcfg.get("checkpoint")
        merged_dir = common.resolve_path(cfg, mcfg.get("merged_dir", "outputs/merged"))
        model_id = cfg["pipeline"]["model"]["id"]
        revision = cfg["pipeline"]["model"]["revision"]

        if not checkpoint:
            if args.dry_run:
                logger.info("[dry-run] peft findings: %s", paramwrapper_findings())
                logger.info("[dry-run] plan: base=%s@%s checkpoint=<UNSET — set "
                            "merge.checkpoint or pass --checkpoint> -> %s",
                            model_id, revision, merged_dir)
                return
            raise PipelineError("No checkpoint: set merge.checkpoint in configs/publish.yaml "
                                "or pass --checkpoint outputs/full/<run>/checkpoint-XXXX")
        checkpoint = common.resolve_path(cfg, checkpoint)
        adapter_file = checkpoint / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise PipelineError(f"{adapter_file} not found — not a LoRA checkpoint dir.")

        fingerprint = common.config_fingerprint({
            "model_id": model_id, "revision": revision, "checkpoint": str(checkpoint),
            "adapter_sha256": common.sha256_file(adapter_file),
            "copy_aux_files": bool(mcfg.get("copy_aux_files", True)),
            "max_shard_size": args.max_shard_size, "dtype": "bfloat16"})

        report_path = merged_dir / "merge_report.json"
        if report_path.exists() and not args.force:
            prev = common.read_json(report_path)
            if prev.get("fingerprint") == fingerprint and prev.get("status") == "ok":
                logger.info("merge already completed for this exact checkpoint+config "
                            "(fingerprint %s) — no-op. Use --force to re-merge.", fingerprint)
                return
            raise PipelineError(
                f"{merged_dir} holds a DIFFERENT merge (fingerprint {prev.get('fingerprint')} "
                f"!= {fingerprint}). Re-run with --force to replace it.")
        if merged_dir.exists() and any(merged_dir.iterdir()) and not report_path.exists() \
                and not args.force:
            raise PipelineError(f"{merged_dir} exists without merge_report.json (interrupted "
                                "or foreign). Re-run with --force to replace it.")

        if args.dry_run:
            logger.info("[dry-run] peft findings: %s", paramwrapper_findings())
            logger.info("[dry-run] plan: base=%s@%s checkpoint=%s -> %s (dtype=bfloat16, "
                        "max_shard_size=%s, copy_aux=%s, fingerprint=%s)", model_id, revision,
                        checkpoint, merged_dir, args.max_shard_size,
                        mcfg.get("copy_aux_files", True), fingerprint)
            logger.info("[dry-run] disk free: %.0f GB (need %s GB); RAM available: %.0f GB "
                        "(need %.0f GB)", shutil.disk_usage(common.project_root()).free / 1e9,
                        cfg["pipeline"]["runtime"]["min_free_disk_gb"].get("merge"),
                        ram_available_gb(), args.min_free_ram_gb)
            return

        common.assert_free_disk(cfg, "merge", logger)
        avail = ram_available_gb()
        if avail < args.min_free_ram_gb:
            raise PipelineError(
                f"Only {avail:.0f} GB RAM available, CPU merge needs ~{args.min_free_ram_gb:.0f} "
                "GB (52 GB bf16 model + save buffers). Free RAM or lower --min-free-ram-gb.")
        logger.info("RAM available: %.0f GB (>= %.0f GB required)", avail, args.min_free_ram_gb)
        logger.info(paramwrapper_findings())

        from huggingface_hub import snapshot_download

        logger.info("resolving base snapshot %s@%s (downloads ~52 GB if not cached)...",
                    model_id, revision)
        base_dir = Path(snapshot_download(model_id, revision=revision))

        work_dir = merged_dir.parent / (merged_dir.name + ".partial")
        if work_dir.exists():
            logger.warning("removing stale partial merge at %s", work_dir)
            shutil.rmtree(work_dir)
        if merged_dir.exists() and args.force:
            logger.warning("--force: removing existing %s", merged_dir)
            shutil.rmtree(merged_dir)

        result = run_merge(base_dir, checkpoint, work_dir, logger, dtype_name="bfloat16",
                           max_shard_size=args.max_shard_size,
                           copy_aux=bool(mcfg.get("copy_aux_files", True)),
                           allow_no_target_parameters=args.allow_no_target_parameters)
        report = {"status": "ok", "fingerprint": fingerprint, "created_utc": common.utc_stamp(),
                  "base_model": model_id, "base_revision": revision, "base_path": str(base_dir),
                  "checkpoint": str(checkpoint), "adapter_sha256": common.sha256_file(adapter_file),
                  "dtype": "bfloat16", "peft_paramwrapper_findings": paramwrapper_findings(),
                  **result}
        common.write_json_atomic(work_dir / "merge_report.json", report)
        merged_dir.parent.mkdir(parents=True, exist_ok=True)
        work_dir.rename(merged_dir)
        logger.info("MERGE OK -> %s (report: %s)", merged_dir, merged_dir / "merge_report.json")
    except PipelineError as exc:
        common.die(logger, str(exc))


if __name__ == "__main__":
    main()
