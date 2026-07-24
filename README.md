# Keural Orbit — Gemma 4 fine-tune

Fine-tune **google/gemma-4-26B-A4B-it** (Gemma 4 — 26B total / ~4B active MoE,
Image-Text-to-Text, 256K context) into **Keural Orbit**, MKD's Korean-specialized model.

**Goal:** keep the base model's benchmark performance on every axis (no quality loss) and
improve **only** Korean understanding & response.

This is a **self-contained project** — its own scripts, configs, and dataset, independent of
the Keural Nova (Qwen) pipeline, so it can be pushed to GitHub on its own. Ground truth for
the architecture and LoRA recipe is **[`VERIFIED-FACTS-ORBIT.md`](VERIFIED-FACTS-ORBIT.md)**
(all hands-on verified on CPU, GPUs untouched).

## Layout

```
Keural Orbit/
├── README.md                    ← this file
├── VERIFIED-FACTS-ORBIT.md      ← verified architecture + LoRA config (read first)
├── requirements.txt             ← load-bearing version pins
├── .gitignore
├── configs/
│   ├── pipeline.yaml            ← shared: paths, model id, publish repo, guard thresholds
│   ├── train_orbit_pilot.yaml   ← pilot SFT (~5k rows, fast checkpoints)
│   ├── train_orbit_full.yaml    ← full SFT (1 epoch, LR 5e-5)
│   ├── eval.yaml                ← lm_eval suites (KMMLU/HAE-RAE/KoBEST/MMLU/GSM8K/HumanEval)
│   └── publish.yaml             ← merge + model-card + upload
├── scripts/
│   ├── common.py                ← shared utils (env guards, GPU checks, logging)
│   ├── verify_lora_targets.py   ← Gemma-4 LoRA safety gate (CPU/meta; --pre & --checkpoint)
│   ├── build_dataset.py         ← rebuild the dataset from public sources (CPU)
│   ├── train.py                 ← ms-swift LoRA SFT launcher (runs the gate automatically)
│   ├── merge_lora.py            ← CPU LoRA merge (fused-expert-aware, byte-verified)
│   ├── run_eval.py              ← baseline / post-train evaluation
│   ├── compare_evals.py         ← baseline-vs-Orbit comparison
│   ├── make_model_card.py       ← Gemma-attributed model card
│   └── publish.py               ← HF upload
└── data/processed/
    ├── train.jsonl              ← 144,812 rows (gitignored — regenerate via build_dataset.py)
    ├── eval.jsonl               ← 1,462 rows (committed)
    ├── pilot.jsonl              ← 5,000 rows (committed)
    └── dataset_manifest.json    ← per-source counts (committed)
```

## Verified LoRA recipe (in the train configs)

- `target_regex` scopes LoRA to the **language model only** — attention (`q/k/v/o_proj`) +
  dense MLP (`gate/up/down_proj`) on all 30 layers. A plain module list crashes on the
  vision tower's `Gemma4ClippableLinear` (VERIFIED-FACTS-ORBIT §3).
- `target_parameters: [gate_up_proj, down_proj]` — the 128 fused routed experts per layer.
- `lora_dropout: 0.0` (required by peft ParamWrapper), rank 32 / α 64, **1 epoch, LR 5e-5**,
  bf16, ZeRO-2, TP=2. **Frozen:** router, vision tower, tied embeddings/lm_head.

## Status

Everything is set up and **verified on CPU** (GPUs/gateway untouched):

| Piece | Status |
|---|---|
| Base model access | ✅ transformers 5.12.1 + ms-swift 4.4.0 both support `gemma4` |
| Architecture + LoRA config | ✅ meta-device peft attach, 0 violations |
| Safety gate (`--pre`) | ✅ PASSES on the real 30-layer model (30/30 experts/attn/mlp) |
| Merge path | ✅ `merge_lora.py --self-test` PASSES end-to-end on a tiny Gemma4 fixture |
| Dataset | ✅ 146,274 rows present in `data/processed/` |
| **Actual training** | ⏳ blocked only on **free GPUs** (Nova gateway holds both H200s) + model download (~52 GB) |

## Run order

```bash
cd "/home/work/keural MoE Pro/Keural Orbit"
source /home/work/.keural_venv/bin/activate      # or your venv
export PYTHONPATH= PYTHONNOUSERSITE=1             # keep ~/.local from shadowing the pins
```

**0. Free the GPUs** (stop the Nova gateway): `pkill -f vllm.entrypoints; pkill -f gateway.py; pkill -f cloudflared`

**1. Baseline-eval stock Gemma 4** — the only proof Korean improved / general held:
```bash
python scripts/run_eval.py --model google/gemma-4-26B-A4B-it --tag gemma4-base
```

**2. Pilot, then full** (train.py runs the Gemma gate `--pre` before launching and re-verifies
the saved adapter after — no manual gate step needed):
```bash
python scripts/train.py --config configs/train_orbit_pilot.yaml   # ~5k rows, sanity
python scripts/train.py --config configs/train_orbit_full.yaml    # full run (tmux)
```

**3. Merge → eval → compare:**
```bash
python scripts/merge_lora.py --checkpoint outputs/full/<run>/checkpoint-XXXX
python scripts/run_eval.py --model outputs/merged/Keural-Orbit-26B-A4B --tag keural-orbit
python scripts/compare_evals.py --tag gemma4-base --tag2 keural-orbit
```

**4. Publish:**
```bash
python scripts/make_model_card.py --config configs/publish.yaml
python scripts/publish.py --config configs/publish.yaml
```

CPU-only sanity checks you can run **right now** (no GPU):
```bash
python scripts/verify_lora_targets.py --pre --train-config configs/train_orbit_pilot.yaml
python scripts/merge_lora.py --self-test
```

## How this differs from the Keural Nova (Qwen) pipeline

- LoRA scoped via **`target_regex`**, not a module list (vision `Gemma4ClippableLinear`).
- **No MTP block** (Gemma 4 has none) → the merge is simpler than Qwen's, but **tied
  embeddings** are preserved (no separate `lm_head.weight` is written).
- Router is `router.proj` (+ scale), not Qwen's `mlp.gate`; template is `gemma4`.

## Honest goal note

SFT preserves + sharpens; it does not add knowledge. Realistic outcome: **match Gemma 4 on
general/knowledge benchmarks (±noise), improve Korean fluency/format/persona** — and you must
**measure** it (step 1 baseline vs step 3 compare) before shipping, not assume it. Context stays
256K. Vision is present but frozen/untested by this fine-tune.
