#!/usr/bin/env python
"""Stage 7b — generate the HF model card (README.md) for the merged Keural model.

Design rules:
* NEVER invent numbers.  Every auto-populated cell (dataset counts, eval scores,
  training hyperparameters) comes from a pipeline artifact
  (dataset_manifest.json / results/<tag>/<ts>/summary.json / train yaml /
  run_metadata.json).  A missing artifact degrades to a "pending" placeholder
  with a WARNING — the card generation itself never crashes on missing inputs,
  because the card is regenerated (cheaply) as artifacts appear.
* The AI Hub license exclusions are stated explicitly (guide §2): shipping a
  commercial model quietly trained on non-commercial AI Hub data would be a
  legal, not just technical, silent failure.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import common
from common import PipelineError

DEFAULT_CONFIG = "configs/publish.yaml"
PENDING = "*pending*"

# Guide §2 — excluded for licensing reasons; listed on the card on purpose.
AIHUB_EXCLUSIONS = [
    ("`keural-SFT` rows tagged `aihub_multisession_sci`", "AI Hub Terms — non-commercial research only"),
    ("`keural-SFT` rows tagged `aihub_multisession_social`", "AI Hub Terms — non-commercial research only"),
    ("`keural-SFT` rows tagged `aihub_expert_qa`", "AI Hub Terms — non-commercial research only"),
    ("`mkd-chanwoo/aihub_conversation` (standalone dataset)", "AI Hub Terms — non-commercial research only"),
    ("any row whose `license` field matches `nc` / `non-commercial` / `aihub`", "belt-and-braces license filter"),
]

# Preferred lm_eval metric keys, most specific first.
METRIC_PREFERENCE = [
    "acc,none", "acc_norm,none", "exact_match,strict-match", "exact_match,none",
    "pass@1", "acc", "acc_norm", "exact_match", "score",
]


# ---------------------------------------------------------------- input loading


def load_json_or_none(path: Path | None, what: str, logger):
    if path is None:
        logger.warning("%s not provided — card will show %s cells", what, PENDING)
        return None
    if not path.exists():
        logger.warning("%s not found at %s — card will show %s cells", what, path, PENDING)
        return None
    try:
        return common.read_json(path)
    except Exception as exc:  # malformed artifact => pending, not crash
        logger.warning("%s at %s unreadable (%s) — card will show %s cells",
                       what, path, exc, PENDING)
        return None


def find_latest_summaries(results_dir: Path, logger) -> tuple[Path | None, Path | None]:
    """Locate the newest baseline / candidate summary.json under results/.

    Convention (eval.yaml): results/<model_tag>/<timestamp>/summary.json.
    Heuristic: tags containing 'keural' are candidate runs, everything else
    (e.g. the stock Qwen tag) is baseline.  Ambiguity resolves to newest mtime.
    """
    baseline, candidate = None, None
    if not results_dir.exists():
        logger.warning("results dir %s missing — eval tables will be %s", results_dir, PENDING)
        return None, None
    for summary in sorted(results_dir.glob("*/*/summary.json"),
                          key=lambda p: p.stat().st_mtime):
        tag = summary.relative_to(results_dir).parts[0].lower()
        if "keural" in tag:
            candidate = summary
        else:
            baseline = summary
    logger.info("auto-discovered summaries: baseline=%s candidate=%s", baseline, candidate)
    return baseline, candidate


def normalize_summary(data) -> dict[str, dict]:
    """Coerce a summary.json (schema owned by the eval component) into
    {task: {metric: value}} without assuming its exact layout."""
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("results"), dict):
        data = data["results"]
    out: dict[str, dict] = {}
    for task, val in data.items():
        if isinstance(val, dict):
            metrics = {k: v for k, v in val.items()
                       if isinstance(v, (int, float)) and not k.startswith("alias")}
            if metrics:
                out[task] = metrics
        elif isinstance(val, (int, float)):
            out[task] = {"score": val}
    return out


def pick_metric(metrics: dict) -> tuple[str, float] | None:
    for key in METRIC_PREFERENCE:
        if key in metrics:
            return key.split(",")[0], metrics[key]
    for key, val in sorted(metrics.items()):
        if "stderr" not in key:
            return key.split(",")[0], val
    return None


def manifest_rows(manifest, logger) -> list[dict] | None:
    """Flexible parse of dataset_manifest.json (schema owned by the data stage)."""
    if manifest is None:
        return None
    src = manifest.get("sources", manifest) if isinstance(manifest, dict) else manifest
    items: list[dict] = []
    if isinstance(src, dict):
        for name, val in src.items():
            items.append({"name": name, **val} if isinstance(val, dict)
                         else {"name": name, "count": val})
    elif isinstance(src, list):
        items = [i for i in src if isinstance(i, dict)]
    if not items:
        logger.warning("dataset manifest has no recognizable per-source entries — "
                       "training-data table will be %s", PENDING)
        return None
    rows = []
    for item in items:
        rows.append({
            "name": str(item.get("name") or item.get("source") or item.get("source_name") or "?"),
            "repo": str(item.get("repo") or item.get("dataset") or item.get("path") or "—"),
            "count": next((item[k] for k in
                           ("sampled", "n_sampled", "count", "rows", "n", "num_examples",
                            "examples") if isinstance(item.get(k), int)), None),
            "license": str(item.get("license") or "—"),
        })
    return rows


# --------------------------------------------------------------- card rendering


def md_escape(val) -> str:
    return str(val).replace("|", "\\|")


def render_eval_table(baseline: dict[str, dict], candidate: dict[str, dict]) -> str:
    tasks = sorted(set(baseline) | set(candidate))
    if not tasks:
        return ("| benchmark | base | Keural |\n|---|---|---|\n"
                f"| KMMLU / HAE-RAE / KoBEST / MMLU / GSM8K / HumanEval | {PENDING} | {PENDING} |\n")
    lines = ["| benchmark | metric | base | Keural |", "|---|---|---:|---:|"]
    for task in tasks:
        b, c = baseline.get(task), candidate.get(task)
        metric = None
        for side in (c, b):
            if side:
                picked = pick_metric(side)
                if picked:
                    metric = picked[0]
                    break

        def cell(side):
            if not side:
                return PENDING
            picked = pick_metric(side)
            return f"{picked[1]:.4f}" if picked else PENDING

        lines.append(f"| {md_escape(task)} | {metric or '—'} | {cell(b)} | {cell(c)} |")
    return "\n".join(lines) + "\n"


def render_data_table(rows: list[dict] | None) -> str:
    if not rows:
        return ("| source | repo | examples | license |\n|---|---|---:|---|\n"
                f"| {PENDING} — dataset_manifest.json not available yet | | | |\n")
    lines = ["| source | repo | examples | license |", "|---|---|---:|---|"]
    total = 0
    for r in rows:
        count = r["count"]
        total += count or 0
        lines.append(f"| {md_escape(r['name'])} | {md_escape(r['repo'])} | "
                     f"{count if count is not None else PENDING} | {md_escape(r['license'])} |")
    if total:
        lines.append(f"| **total** | | **{total}** | |")
    return "\n".join(lines) + "\n"


def render_train_config(train_cfg: dict | None, run_meta: dict | None) -> str:
    if not train_cfg:
        return f"Training configuration: {PENDING}\n"
    sa = train_cfg.get("swift_args", {})
    rows = [
        ("method", "LoRA (ms-swift `swift sft`)"),
        ("LoRA rank / alpha / dropout",
         f"{sa.get('lora_rank')} / {sa.get('lora_alpha')} / {sa.get('lora_dropout')}"),
        ("target_modules (nn.Linear)", "`" + " ".join(map(str, sa.get("target_modules", []))) + "`"),
        ("target_parameters (fused routed experts)",
         "`" + " ".join(map(str, sa.get("target_parameters", []))) + "`"),
        ("frozen", "router (`mlp.gate`), `embed_tokens`, `lm_head`, vision tower (`freeze_vit`)"),
        ("learning rate / schedule",
         f"{sa.get('learning_rate')} / {sa.get('lr_scheduler_type')} "
         f"(warmup_ratio {sa.get('warmup_ratio')})"),
        ("epochs", sa.get("num_train_epochs")),
        ("effective batch size",
         f"{sa.get('per_device_train_batch_size')} x 2 GPUs x "
         f"{sa.get('gradient_accumulation_steps')} grad-accum"),
        ("max sequence length", sa.get("max_length")),
        ("precision / parallelism",
         f"{sa.get('torch_dtype')} + deepspeed {sa.get('deepspeed')} on 2x H200"),
        ("loss masking",
         f"assistant tokens only; `--loss_scale {sa.get('loss_scale')}` zero-masks the "
         "empty `<think></think>` scaffold"),
    ]
    if run_meta:
        for key in ("run_name", "checkpoint", "global_step", "train_loss", "started", "finished"):
            if key in run_meta:
                rows.append((f"run {key}", run_meta[key]))
    lines = ["| setting | value |", "|---|---|"]
    lines += [f"| {k} | {md_escape(v)} |" for k, v in rows]
    return "\n".join(lines) + "\n"


def build_card(cfg: dict, args, logger) -> str:
    mc = cfg.get("model_card") or {}
    name = mc.get("name", "Keural-Orbit-26B-A4B")
    base_model = mc.get("base_model", "google/gemma-4-26B-A4B-it")
    languages = mc.get("languages", ["ko", "en"])
    license_id = mc.get("license", "apache-2.0")
    repo = (cfg.get("upload") or {}).get("repo", f"mkd-hossain/{name}")
    base_revision = cfg["pipeline"]["model"].get("revision", "main")

    manifest_path = (Path(args.manifest) if args.manifest
                     else common.resolve_path(cfg, "data/processed/dataset_manifest.json"))
    manifest = load_json_or_none(manifest_path, "dataset manifest", logger)
    data_rows = manifest_rows(manifest, logger)

    results_dir = common.resolve_path(cfg, cfg["pipeline"]["paths"]["results_dir"])
    baseline_path = Path(args.baseline) if args.baseline else None
    candidate_path = Path(args.candidate) if args.candidate else None
    if baseline_path is None and candidate_path is None:
        baseline_path, candidate_path = find_latest_summaries(results_dir, logger)
    baseline = normalize_summary(load_json_or_none(baseline_path, "baseline summary", logger))
    candidate = normalize_summary(load_json_or_none(candidate_path, "candidate summary", logger))

    train_cfg = None
    train_cfg_path = common.project_root() / args.train_config
    if train_cfg_path.exists():
        train_cfg = common.load_yaml(args.train_config)
    else:
        logger.warning("train config %s missing — training table will be %s",
                       train_cfg_path, PENDING)
    run_meta = None
    if args.run_dir:
        run_meta = load_json_or_none(Path(args.run_dir) / "run_metadata.json",
                                     "run metadata", logger)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tags = ["lora", "moe", "korean", "keural", "gemma4"]
    fm = ["---", f"license: {license_id}", f"base_model: {base_model}",
          "language:"] + [f"- {lang}" for lang in languages] + \
         ["library_name: transformers", "pipeline_tag: image-text-to-text", "tags:"] + \
         [f"- {t}" for t in tags] + ["---"]

    body = f"""
# {name}

**{name}** is [MKD](https://huggingface.co/mkd-hossain)'s Korean-focused assistant model:
a LoRA SFT fine-tune of [`{base_model}`](https://huggingface.co/{base_model})
(revision `{base_revision}`), merged back into full weights. The goal is to **preserve the
base model's benchmarks on every axis and improve only Korean** understanding and response.

This release deliberately preserves the whole base architecture:

- **Mixture-of-Experts** (128 routed experts, top-8, ~4B active of 26B): the fused
  `experts.gate_up_proj` / `experts.down_proj` `nn.Parameter` tensors were adapted via
  peft `target_parameters`, alongside the dense MLP and attention projections on all 30
  layers; the **router stayed frozen**.
- **Hybrid attention**: sliding-window + full attention (Gemma 4's 5-in-6 layout) — LoRA
  applied to `q/k/v/o_proj` throughout (full-attention layers share K=V, no separate v_proj).
- **Multimodal (Image-Text-to-Text)**: the vision tower is shipped **unchanged** (frozen
  during SFT), together with the image preprocessor files.
- **Tied embeddings** (`tie_word_embeddings: true`) are preserved on merge; the router,
  embeddings and vision tower are byte-identical to the base (see `merge_report.json`).

## Training data

Sources (auto-populated from the pipeline's `dataset_manifest.json`):

{render_data_table(data_rows)}
Roughly 85% Korean / 15% English-replay instruction data, all ChatML multi-turn,
max 8,192 tokens per example, deduplicated and shuffled.

### Explicitly excluded data (licensing)

The following sources were **excluded** because their AI Hub terms restrict
commercial use (guide §2 — worst-case reading, on purpose):

{chr(10).join(f"- {what} — {why}" for what, why in AIHUB_EXCLUSIONS)}

## Evaluation

{render_eval_table(baseline, candidate)}
Baseline = stock `{base_model}` run through the identical lm_eval harness
(see `results/` artifacts of the pipeline). Cells marked {PENDING} mean the
corresponding eval run has not been executed/registered yet — they are never guessed.

## Training configuration

{render_train_config(train_cfg, run_meta)}
## Usage

```python
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

repo = "{repo}"
processor = AutoProcessor.from_pretrained(repo)
model = Gemma4ForConditionalGeneration.from_pretrained(
    repo, dtype="bfloat16", device_map="auto"
)

messages = [{{"role": "user", "content": [{{"type": "text", "text": "한국의 수도는 어디인가요?"}}]}}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
out = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Serving with vLLM (2x GPU tensor parallel):

```bash
vllm serve {repo} --tensor-parallel-size 2 --max-model-len 8192
```

## Limitations & notes

- Korean-focused: English capability relies on a small replay mix; for most non-Korean
  tasks expect the base model's behavior (the aim is parity, not regression).
- The vision tower is frozen and unevaluated in this release: multimodal behavior is
  inherited from the base model.
- Router, embeddings and (tied) `lm_head` are byte-identical to the base model.
- Context length matches the base (256K); no YaRN extension is applied by this fine-tune.

## Provenance

- Base: `{base_model}` @ `{base_revision}` (Gemma Terms of Use — Google DeepMind).
- Merge: CPU weight-level LoRA merge with byte-level verification (fused-expert-aware,
  tied-embedding preserving) — see `merge_report.json` in this repo.
- Card generated {now} by the Keural Orbit pipeline (`make_model_card.py`).
"""
    return "\n".join(fm) + "\n" + body.lstrip("\n")


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--merged-dir", default=None,
                        help="where README.md is written (default: cfg merge.merged_dir)")
    parser.add_argument("--out", default=None,
                        help="explicit output file path (overrides --merged-dir)")
    parser.add_argument("--manifest", default=None, help="dataset_manifest.json path")
    parser.add_argument("--baseline", default=None, help="baseline summary.json path")
    parser.add_argument("--candidate", default=None, help="candidate summary.json path")
    parser.add_argument("--run-dir", default=None,
                        help="training run dir containing run_metadata.json")
    parser.add_argument("--train-config", default="configs/train_full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("model_card", cfg)
    common.stage_banner(logger, "make_model_card", args.config)
    common.guard_environment(logger)

    try:
        if args.out:
            out_path = Path(args.out)
        else:
            merged_dir = (Path(args.merged_dir) if args.merged_dir
                          else common.resolve_path(cfg, (cfg.get("merge") or {})
                                                   .get("merged_dir", "outputs/merged")))
            out_path = merged_dir / "README.md"

        card = build_card(cfg, args, logger)
        fingerprint = common.config_fingerprint(card)
        stamped = card + f"\n<!-- keural-card-fingerprint: {fingerprint} -->\n"

        if out_path.exists() and not args.force:
            if f"keural-card-fingerprint: {fingerprint}" in out_path.read_text(encoding="utf-8"):
                logger.info("card at %s already up to date (fingerprint %s) — no-op. "
                            "Use --force to rewrite.", out_path, fingerprint)
                return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".md.tmp")
        tmp.write_text(stamped, encoding="utf-8")
        tmp.replace(out_path)
        logger.info("model card written: %s (%d bytes, fingerprint %s)",
                    out_path, len(stamped.encode()), fingerprint)
    except PipelineError as exc:
        common.die(logger, str(exc))


if __name__ == "__main__":
    main()
