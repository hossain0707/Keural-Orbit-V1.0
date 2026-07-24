#!/usr/bin/env python
"""Stages 2 & 6 — evaluation harness (baseline + post-training).

Runs lm_eval 0.4.12 benchmark suites (Korean: kmmlu/haerae/kobest; general
regression tripwire: mmlu/gsm8k/humaneval — guide §1) plus our own domain eval
(evals/domain/*.jsonl) against either the stock base model or a local (merged)
checkpoint, and writes a normalized, versioned summary under results/<tag>/<stamp>/.

Design notes (WHY):
- One lm_eval subprocess PER SUITE GROUP, not one for everything: a crash in one
  suite (e.g. a humaneval sandbox issue) must not lose hours of completed Korean
  results. Each group's raw output lands in its own subdir and is harvested
  independently; failed groups are reported and the stage exits non-zero.
- lm_eval facts below were verified hands-on against the installed 0.4.12
  (2026-07-08), NOT guessed: `kmmlu`/`haerae`/`kobest`/`mmlu`/`gsm8k`/`humaneval`
  all resolve in the task registry (the first four as groups); results JSON is
  `{"results": {<task>: {"<metric>,<filter>": val, "<metric>_stderr,<filter>":
  val|"N/A", "name":..., "alias":..., "sample_len":...}}}`, written to
  `<output_path>/<model_name_sanitized>/results_<isodate>.json`.
- Benchmark generation settings are left to each task's own config (gsm8k pins
  temperature 0, humaneval pins do_sample=false — verified in the installed task
  yamls), so published-comparable defaults are preserved; cfg generation
  .temperature applies to the DOMAIN suite generation only.
- GPU safety: NEVER initializes CUDA in-process for anything CPU-testable.
  All GPU work (lm_eval subprocess, domain generation) is gated behind
  common.assert_gpus_free() unless --allow-busy-gpus, and --dry-run never
  touches GPUs at all (VERIFIED-FACTS §1: another team's job owns them).
- Idempotency: a config fingerprint (model ref + backend + suites + smoke +
  eval cfg + domain-file SHAs + local-model config.json SHA) is stamped into
  run_state.json/summary.json. Re-running a completed identical eval is a no-op
  unless --force. A killed/partial run with the same fingerprint is RESUMED in
  its own run dir (completed suite groups are skipped); a different fingerprint
  always gets a fresh run dir — existing run dirs are never overwritten.
- The project path contains a SPACE: every subprocess uses arg lists via
  common.run_logged (no shell string concatenation anywhere).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import PipelineError  # noqa: E402

DEFAULT_CONFIG = "configs/eval.yaml"
BENCH_SUITES = ("korean", "general")
ALL_SUITES = ("korean", "general", "domain")
# lm_eval per-task result entries mix metrics with metadata; keep only metrics.
NON_METRIC_KEYS = {"name", "alias", "sample_len", "samples"}
HUB_ID_RE = re.compile(r"[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(@[A-Za-z0-9_.\-]+)?")
TAG_RE = re.compile(r"[A-Za-z0-9._\-]+")


# ------------------------------------------------------------------ CLI / model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="stage config yaml (pipeline.yaml is merged automatically)")
    p.add_argument("--model", default="base",
                   help="'base' (pipeline model.id@revision), a local model dir "
                        "(e.g. the merged model), or an explicit org/name[@rev] hub id")
    p.add_argument("--tag", default=None,
                   help="results/<tag>/ dir name (default: 'base' or model dir basename)")
    p.add_argument("--suite", choices=["all", *ALL_SUITES], default="all",
                   help="which suite group(s) to run")
    p.add_argument("--backend", choices=["vllm", "hf"], default=None,
                   help="inference backend (default: cfg backend.default)")
    p.add_argument("--smoke", action="store_true",
                   help="fast smoke test: apply cfg smoke_limit as lm_eval --limit "
                        "(and cap domain examples)")
    p.add_argument("--allow-busy-gpus", action="store_true",
                   help="skip the busy-GPU guard (DANGEROUS: may OOM another team's job)")
    p.add_argument("--log-samples", action="store_true",
                   help="pass --log_samples to lm_eval (per-sample dumps are huge; off by default)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the exact lm_eval commands + domain plan and exit "
                        "(no GPUs touched, nothing written under results/)")
    p.add_argument("--force", action="store_true",
                   help="re-run even if an identical completed run exists")
    return p.parse_args(argv)


def resolve_model(model_arg: str, cfg: dict) -> tuple[str, str | None, str, bool]:
    """-> (model_ref, revision_or_None, default_tag, is_hub_id).

    Strict on purpose: a typo'd path silently falling through to a hub download
    would evaluate the WRONG model and poison the baseline-vs-trained comparison.
    """
    if model_arg == "base":
        m = cfg["pipeline"]["model"]
        return m["id"], m.get("revision"), "base", True
    p = Path(model_arg).expanduser()
    if p.exists():
        p = p.resolve()
        if not p.is_dir() or not (p / "config.json").exists():
            raise PipelineError(
                f"--model {p} exists but is not a model dir (no config.json). "
                "Point at the directory that holds config.json + safetensors shards."
            )
        return str(p), None, p.name, False
    if HUB_ID_RE.fullmatch(model_arg):
        ref, _, rev = model_arg.partition("@")
        return ref, (rev or None), ref.split("/")[-1], True
    raise PipelineError(
        f"--model {model_arg!r} is neither 'base', an existing local model dir, nor "
        "an org/name[@revision] hub id. For the merged model pass its directory, e.g. "
        '"outputs/merged/Keural-35B-A3B".'
    )


def selected_suites(suite_arg: str) -> list[str]:
    return list(ALL_SUITES) if suite_arg == "all" else [suite_arg]


# ------------------------------------------------------------------ lm_eval cmd


def build_model_args(model_ref: str, revision: str | None, backend: str,
                     cfg: dict, seed: int) -> str:
    """Comma-joined lm_eval --model_args (passed as ONE argv token, so the space
    in the project path is safe; a comma in the path would break lm_eval's
    key=val,key2=val2 parser, hence the guard)."""
    if "," in model_ref:
        raise PipelineError(
            f"model path/id contains a comma, which breaks lm_eval --model_args "
            f"parsing: {model_ref!r}. Move/rename the model dir."
        )
    parts = [f"pretrained={model_ref}"]
    if backend == "vllm":
        for k, v in (cfg["backend"].get("vllm") or {}).items():
            parts.append(f"{k}={v}")
        if revision:
            parts.append(f"revision={revision}")
        # vllm sampling seed for determinism (lm_eval --seed does not reach vLLM).
        parts.append(f"seed={seed}")
    elif backend == "hf":
        extra = (cfg["backend"].get("hf") or {}).get("model_args_extra") or ""
        if extra:
            parts.append(extra)
        if revision:
            parts.append(f"revision={revision}")
    else:  # pragma: no cover - argparse restricts choices
        raise PipelineError(f"unknown backend {backend!r}")
    return ",".join(parts)


def build_lm_eval_cmd(suite: str, tasks: list[str], run_dir: Path, model_ref: str,
                      revision: str | None, backend: str, cfg: dict, seed: int,
                      smoke: bool, log_samples: bool) -> tuple[list[str], dict]:
    """-> (argv, extra_env). One command per suite group (crash isolation).

    Invoked as `venv_python() -m lm_eval`, NOT the venv/bin/lm_eval console
    script — that wrapper's shebang hardcodes the real (space-containing)
    venv path, which breaks flashinfer's CUDA JIT compilation. See
    common.venv_python() for the full explanation.
    """
    cmd = [
        str(common.venv_python()), "-m", "lm_eval", "run",
        "--model", backend,
        "--model_args", build_model_args(model_ref, revision, backend, cfg, seed),
        "--tasks", *tasks,
        "--batch_size", str(cfg.get("batch_size", "auto")),
        "--seed", str(seed),
        "--output_path", str(run_dir / "raw" / suite),
    ]
    if cfg.get("num_fewshot") is not None:
        cmd += ["--num_fewshot", str(cfg["num_fewshot"])]
    if smoke:
        cmd += ["--limit", str(cfg["smoke_limit"])]
    if log_samples:
        cmd += ["--log_samples"]
    extra_env: dict = {}
    if backend == "vllm":
        # DRIVER-COMPAT (verified 2026-07-23, driver 570.86.10 / CUDA 12.8). vLLM
        # 0.24.0's newest CUDA fast-paths all want CUDA 12.9 and die with
        # `cudaErrorInsufficientDriver`. Three of them bite in sequence, so all are
        # disabled here:
        #   1. DeepGEMM MoE kernels           -> VLLM_USE_DEEP_GEMM=0
        #   2. CUDA-graph capture             -> enforce_eager (set in eval.yaml)
        #   3. FlashAttention v4 CUTLASS-DSL  -> force Triton attention backend.
        # Gemma 4 has heterogeneous head dims (256/512) so vLLM auto-picks FA4,
        # whose cutlass-dsl kernels need driver>=12.9; Triton JIT-compiles for the
        # actual installed driver instead. (The Qwen gateway never hit FA4.)
        extra_env["VLLM_USE_DEEP_GEMM"] = "0"
        extra_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        extra_env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    if "humaneval" in tasks:
        # Verified requirement (lm_eval 0.4.12 + VERIFIED-FACTS §6): code eval
        # executes generated code, so it must be armed explicitly.
        cmd += ["--confirm_run_unsafe_code"]
        extra_env["HF_ALLOW_CODE_EVAL"] = "1"
    return cmd, extra_env


def harvest_suite_results(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Parse the newest lm_eval results_*.json under raw_dir into {task: {metric: value}}.

    lm_eval writes <raw_dir>/<model_name_sanitized>/results_<isodate>.json (verified
    on 0.4.12); ISO date stamps sort lexicographically, so the last glob hit is the
    newest. Non-numeric entries ("N/A" stderr, name/alias metadata) are dropped so
    summary.json stays purely numeric per metric.
    """
    files = sorted(raw_dir.glob("**/results_*.json"))
    if not files:
        return {}
    data = common.read_json(files[-1])
    out: dict[str, dict[str, float]] = {}
    for task, entry in (data.get("results") or {}).items():
        metrics = {k: v for k, v in entry.items()
                   if k not in NON_METRIC_KEYS and isinstance(v, (int, float))
                   and not isinstance(v, bool)}
        if metrics:
            out[task] = metrics
    return out


# ------------------------------------------------------------------ domain suite


def scan_domain_files(domain_dir: Path) -> list[Path]:
    """Only *.jsonl participates — example.jsonl.disabled is excluded by design."""
    if not domain_dir.exists():
        return []
    return sorted(p for p in domain_dir.glob("*.jsonl") if p.is_file())


def load_domain_examples(files: list[Path]) -> list[dict]:
    """Strict schema validation ({"id","prompt","reference","rubric"?}).

    Hard-fails with file:line on any violation: a silently-mangled domain set
    would produce garbage comparisons — the exact failure mode this pipeline
    exists to prevent.
    """
    examples: list[dict] = []
    seen_ids: dict[str, str] = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                where = f"{f}:{lineno}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise PipelineError(f"domain eval: invalid JSON at {where}: {e}") from e
                if not isinstance(row, dict):
                    raise PipelineError(f"domain eval: row is not an object at {where}")
                for key in ("id", "prompt", "reference"):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise PipelineError(
                            f"domain eval: missing/empty required field {key!r} at {where} "
                            "(schema: {\"id\",\"prompt\",\"reference\",\"rubric\"?}, all strings)"
                        )
                if "rubric" in row and not isinstance(row["rubric"], str):
                    raise PipelineError(f"domain eval: 'rubric' must be a string at {where}")
                if row["id"] in seen_ids:
                    raise PipelineError(
                        f"domain eval: duplicate id {row['id']!r} at {where} "
                        f"(first seen at {seen_ids[row['id']]})"
                    )
                seen_ids[row["id"]] = where
                examples.append({"id": row["id"], "prompt": row["prompt"],
                                 "reference": row["reference"],
                                 "rubric": row.get("rubric")})
    return examples


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _strip_think(s: str) -> str:
    """Drop a leading <think>...</think> block — the model emits the empty-think
    scaffold on the non-thinking path (guide §3); it must not pollute scoring."""
    return re.sub(r"^\s*<think>.*?</think>\s*", "", s, flags=re.DOTALL)


def score_domain(examples: list[dict], outputs: list[str]) -> tuple[list[dict], dict]:
    """Cheap exact/contains heuristic ONLY + needs_human_or_judge marker.

    Rubric / LLM-judge grading is a documented placeholder (guide §1 asks for it;
    no judge endpoint exists in this environment — see evals/domain/README.md).
    """
    rows, n_exact, n_contains = [], 0, 0
    for ex, out in zip(examples, outputs):
        cleaned = _strip_think(out)
        exact = _normalize(cleaned) == _normalize(ex["reference"])
        contains = _normalize(ex["reference"]) in _normalize(cleaned)
        n_exact += exact
        n_contains += contains
        rows.append({"id": ex["id"], "prompt": ex["prompt"],
                     "reference": ex["reference"], "rubric": ex.get("rubric"),
                     "output": out, "exact": exact, "contains": contains})
    n = len(rows)
    summary = {
        "n_examples": n,
        "exact_match_rate": (n_exact / n) if n else 0.0,
        "contains_rate": (n_contains / n) if n else 0.0,
        "needs_human_or_judge": True,
        "grading": "heuristic exact/contains only; rubric/LLM-judge integration is a "
                   "documented placeholder (guide §1) — no judge endpoint exists here. "
                   "Grade domain_generations.jsonl by hand or via a future judge stage.",
    }
    return rows, summary


def generate_domain_outputs(examples: list[dict], model_ref: str, revision: str | None,
                            backend: str, cfg: dict, seed: int,
                            logger) -> list[str]:
    """GPU path (guarded by assert_gpus_free upstream). Renders each prompt with
    the model's chat template (enable_thinking=False — must match the training
    data's empty-think format, guide §3) and generates deterministically."""
    from transformers import AutoTokenizer  # lazy: keep CPU paths import-free

    tok = AutoTokenizer.from_pretrained(model_ref, revision=revision)
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for ex in examples
    ]
    max_new = int(cfg["domain"]["max_new_tokens"])
    temperature = float(cfg.get("generation", {}).get("temperature", 0.0))
    if backend == "vllm":
        return _generate_vllm(prompts, model_ref, revision, cfg, seed, max_new,
                              temperature, logger)
    return _generate_hf(prompts, model_ref, revision, tok, max_new, temperature, logger)


def _generate_vllm(prompts: list[str], model_ref: str, revision: str | None,
                   cfg: dict, seed: int, max_new: int, temperature: float,
                   logger) -> list[str]:
    from vllm import LLM, SamplingParams  # lazy import: initializes CUDA

    v = dict(cfg["backend"].get("vllm") or {})
    kwargs = dict(
        model=model_ref,
        tensor_parallel_size=int(v.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(v.get("gpu_memory_utilization", 0.85)),
        max_model_len=int(v.get("max_model_len", 8192)),
        dtype=str(v.get("dtype", "bfloat16")),
        seed=seed,
    )
    if revision:
        kwargs["revision"] = revision
    logger.info("domain: loading vllm engine (%s)", model_ref)
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=temperature, max_tokens=max_new)
    outs = llm.generate(prompts, sp)  # preserves input order
    return [o.outputs[0].text for o in outs]


def _generate_hf(prompts: list[str], model_ref: str, revision: str | None,
                 tok, max_new: int, temperature: float, logger) -> list[str]:
    import torch  # lazy import: initializes CUDA on .to/device_map
    from transformers import AutoModelForCausalLM

    logger.info("domain: loading hf model (%s, device_map=auto)", model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref, revision=revision, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    outputs = []
    # do_sample=False when temperature==0: transformers raises on temperature=0
    # with sampling enabled, and greedy is what "deterministic" means anyway.
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=temperature > 0)
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    for i, prompt in enumerate(prompts, 1):
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, **gen_kwargs)
        outputs.append(tok.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True))
        logger.info("domain: generated %d/%d", i, len(prompts))
    return outputs


def run_domain_suite(run_dir: Path, examples: list[dict], n_files: int,
                     model_ref: str, revision: str | None, backend: str,
                     cfg: dict, seed: int, logger):
    """-> domain summary (dict) or the literal skip string. Resumable: an already
    completed generation+score pass in this run dir is reused as-is."""
    gen_path = run_dir / "domain_generations.jsonl"
    sum_path = run_dir / "domain_summary.json"
    if gen_path.exists() and sum_path.exists():
        logger.info("domain: reusing completed generations in %s (resume)", run_dir)
        return common.read_json(sum_path)
    outputs = generate_domain_outputs(examples, model_ref, revision, backend,
                                      cfg, seed, logger)
    if len(outputs) != len(examples):
        raise PipelineError(
            f"domain: got {len(outputs)} generations for {len(examples)} examples — "
            "backend returned a partial batch; not writing misaligned scores."
        )
    rows, summary = score_domain(examples, outputs)
    summary["n_files"] = n_files
    tmp = gen_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(gen_path)
    common.write_json_atomic(sum_path, summary)
    logger.info("domain: %d examples | exact %.1f%% | contains %.1f%% (heuristic only)",
                summary["n_examples"], 100 * summary["exact_match_rate"],
                100 * summary["contains_rate"])
    return summary


# ------------------------------------------------------------------ run dirs / idempotency


def compute_fingerprint(model_ref: str, revision: str | None, backend: str,
                        suites: list[str], smoke: bool, seed: int, cfg: dict,
                        domain_files: list[Path]) -> str:
    """Fingerprint everything that changes eval RESULTS. Domain file SHAs are
    included so adding/editing domain examples invalidates 'completed' status;
    a local model's config.json SHA is included so a rebuilt merge at the same
    path is not mistaken for the already-evaluated one."""
    payload = {
        "model": model_ref,
        "revision": revision,
        "backend": backend,
        "suites": suites,
        "smoke": smoke,
        "seed": seed,
        "eval_cfg": {k: v for k, v in cfg.items() if k != "pipeline"},
        "domain_files": ([[f.name, common.sha256_file(f)] for f in domain_files]
                         if "domain" in suites else None),
    }
    local_cfg = Path(model_ref) / "config.json"
    if revision is None and local_cfg.exists():
        payload["local_model_config_sha"] = common.sha256_file(local_cfg)
    return common.config_fingerprint(payload)


def find_prior_run(tag_dir: Path, fingerprint: str) -> tuple[Path | None, Path | None]:
    """-> (completed_run_dir_or_None, resumable_run_dir_or_None) for this fingerprint."""
    completed = resumable = None
    if not tag_dir.exists():
        return None, None
    for run in sorted((d for d in tag_dir.iterdir() if d.is_dir()), reverse=True):
        state_p, summary_p = run / "run_state.json", run / "summary.json"
        summary = _read_json_or_none(summary_p)
        if summary and summary.get("config_fingerprint") == fingerprint \
                and summary.get("status") == "complete" and completed is None:
            completed = run
        state = _read_json_or_none(state_p)
        if state and state.get("config_fingerprint") == fingerprint and resumable is None \
                and (summary is None or summary.get("status") != "complete"):
            resumable = run
    return completed, resumable


def _read_json_or_none(path: Path):
    if not path.exists():
        return None
    try:
        return common.read_json(path)
    except (json.JSONDecodeError, OSError):
        return None


def make_run_dir(tag_dir: Path) -> Path:
    """Fresh, collision-proof run dir — an existing dir is NEVER overwritten."""
    stamp = common.utc_stamp()
    for i in range(100):
        candidate = tag_dir / (stamp if i == 0 else f"{stamp}-{i + 1}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise PipelineError(f"could not create a unique run dir under {tag_dir}")


# ------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = common.load_config(args.config)
    logger = common.setup_logging("eval", cfg)
    common.stage_banner(logger, "eval", args.config)
    common.guard_environment(logger)
    common.assert_free_disk(cfg, "eval", logger)

    model_ref, revision, default_tag, is_hub = resolve_model(args.model, cfg)
    tag = args.tag or default_tag
    if not TAG_RE.fullmatch(tag):
        raise PipelineError(f"--tag {tag!r} must match {TAG_RE.pattern} (it names a dir)")
    backend = args.backend or cfg["backend"]["default"]
    seed = int(cfg["pipeline"]["seed"])
    suites = selected_suites(args.suite)
    results_dir = common.resolve_path(cfg, cfg["output"]["results_dir"])
    tag_dir = results_dir / tag

    # Domain inventory up-front: validation must fail fast (even in --dry-run),
    # and the file SHAs feed the fingerprint.
    domain_dir = common.resolve_path(cfg, cfg["domain"]["dir"])
    domain_files = scan_domain_files(domain_dir) if "domain" in suites else []
    domain_examples = load_domain_examples(domain_files) if domain_files else []
    if args.smoke and domain_examples:
        domain_examples = domain_examples[: int(cfg["smoke_limit"])]

    fingerprint = compute_fingerprint(model_ref, revision, backend, suites,
                                      args.smoke, seed, cfg, domain_files)
    logger.info("model=%s revision=%s tag=%s backend=%s suites=%s smoke=%s fp=%s",
                model_ref, revision, tag, backend, suites, args.smoke, fingerprint)

    bench_plan = {s: list(cfg["suites"][s]) for s in suites if s in BENCH_SUITES}

    completed, resumable = find_prior_run(tag_dir, fingerprint)
    if completed and not args.force:
        logger.info("identical completed run exists: %s — no-op (use --force to re-run)",
                    completed)
        if not args.dry_run:
            return 0

    # ---------------- dry run: print exact commands, touch nothing ----------------
    if args.dry_run:
        run_dir_preview = tag_dir / f"<{common.utc_stamp()}>"
        print(f"[dry-run] fingerprint={fingerprint}  run dir would be: {run_dir_preview}")
        if completed:
            print(f"[dry-run] NOTE: identical completed run exists at {completed} — "
                  "a real run would no-op without --force")
        elif resumable:
            print(f"[dry-run] NOTE: partial run with same fingerprint at {resumable} — "
                  "a real run would resume it")
        for suite, tasks in bench_plan.items():
            cmd, extra_env = build_lm_eval_cmd(
                suite, tasks, run_dir_preview, model_ref, revision, backend, cfg,
                seed, args.smoke, args.log_samples)
            env_str = " ".join(f"{k}={v}" for k, v in extra_env.items())
            print(f"[dry-run] suite {suite!r}:")
            if env_str:
                print(f"  env: {env_str}")
            print(f"  cmd: {shlex.join(cmd)}")
        if "domain" in suites:
            if not domain_files:
                print(f"[dry-run] suite 'domain': SKIP — no *.jsonl under {domain_dir} "
                      "(WARNING would be logged; summary records "
                      "'skipped (no files)'; exit 0 for the suite)")
            else:
                print(f"[dry-run] suite 'domain': {len(domain_examples)} validated "
                      f"example(s) from {len(domain_files)} file(s): "
                      f"{[f.name for f in domain_files]}")
                print(f"  would generate with backend={backend!r} model={model_ref!r} "
                      f"revision={revision!r} temperature="
                      f"{cfg.get('generation', {}).get('temperature', 0.0)} "
                      f"max_new_tokens={cfg['domain']['max_new_tokens']} "
                      "(chat template, enable_thinking=False)")
        print("[dry-run] done — nothing was executed or written under results/")
        return 0

    # ---------------- GPU guard: only if something actually needs GPUs ----------------
    run_dir = resumable if (resumable and not args.force) else make_run_dir(tag_dir)
    if resumable and run_dir == resumable:
        logger.info("resuming partial run: %s", run_dir)
    logger.info("run dir: %s", run_dir)

    pending_bench = {s: t for s, t in bench_plan.items()
                     if not harvest_suite_results(run_dir / "raw" / s)}
    domain_pending = bool(domain_examples) and not (
        (run_dir / "domain_generations.jsonl").exists()
        and (run_dir / "domain_summary.json").exists())
    if pending_bench or domain_pending:
        if args.allow_busy_gpus:
            logger.warning("--allow-busy-gpus: SKIPPING the busy-GPU guard — you may "
                           "OOM another team's job (VERIFIED-FACTS §1)")
        else:
            common.assert_gpus_free(cfg, logger)

    common.write_json_atomic(run_dir / "run_state.json", {
        "config_fingerprint": fingerprint, "model": model_ref, "revision": revision,
        "tag": tag, "backend": backend, "suites": suites, "smoke": args.smoke,
        "started": common.utc_stamp(),
    })
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # ---------------- benchmark suite groups (isolated subprocesses) ----------------
    all_task_metrics: dict[str, dict] = {}
    task_suites: dict[str, str] = {}
    failed: list[str] = []
    for suite, tasks in bench_plan.items():
        raw_dir = run_dir / "raw" / suite
        existing = harvest_suite_results(raw_dir)
        if existing:
            logger.info("suite %r: results already present (resume) — skipping run", suite)
            harvested = existing
        else:
            cmd, extra_env = build_lm_eval_cmd(
                suite, tasks, run_dir, model_ref, revision, backend, cfg,
                seed, args.smoke, args.log_samples)
            rc = common.run_logged(cmd, logger, run_dir / "logs" / f"lm_eval_{suite}.log",
                                   env=common.child_env(extra_env))
            if rc != 0:
                logger.error("suite %r FAILED (lm_eval exit %d) — continuing with the "
                             "remaining suites; see %s", suite, rc,
                             run_dir / "logs" / f"lm_eval_{suite}.log")
                failed.append(suite)
                continue
            harvested = harvest_suite_results(raw_dir)
            if not harvested:
                logger.error("suite %r: lm_eval exited 0 but produced no results_*.json "
                             "under %s — treating as failure (silent-failure guard)",
                             suite, raw_dir)
                failed.append(suite)
                continue
        for task, metrics in harvested.items():
            all_task_metrics[task] = metrics
            task_suites[task] = suite
        logger.info("suite %r: harvested %d task result(s)", suite, len(harvested))

    # ---------------- domain suite ----------------
    domain_result = None
    if "domain" in suites:
        if not domain_files:
            logger.warning("domain suite: no *.jsonl files under %s — SKIPPING "
                           "(the team has not provided domain examples yet; see "
                           "evals/domain/README.md). Recording it in summary.json.",
                           domain_dir)
            domain_result = "skipped (no files)"
        else:
            try:
                domain_result = run_domain_suite(
                    run_dir, domain_examples, len(domain_files), model_ref, revision,
                    backend, cfg, seed, logger)
            except PipelineError:
                raise
            except Exception as e:  # generation backends raise all sorts — isolate
                logger.error("domain suite FAILED: %s: %s", type(e).__name__, e)
                failed.append("domain")

    # ---------------- summary ----------------
    suites_out: dict = dict(all_task_metrics)
    if domain_result is not None:
        suites_out["domain"] = domain_result
        task_suites["domain"] = "domain"
    status = "complete" if not failed else "partial"
    summary = {
        "model": cfg["pipeline"]["model"]["id"] if args.model == "base" else args.model,
        "tag": tag,
        "revision_or_path": revision if is_hub else model_ref,
        "timestamp": common.utc_stamp(),
        "suites": suites_out,
        "task_suites": task_suites,  # extra: lets compare_evals.py classify tasks
        "config_fingerprint": fingerprint,
        "backend": backend,
        "smoke": bool(args.smoke),
        "status": status,
        "failed_suites": failed,
        "seed": seed,
        "run_dir": str(run_dir),
    }
    common.write_json_atomic(run_dir / "summary.json", summary)
    logger.info("summary written: %s", run_dir / "summary.json")

    if failed:
        common.die(logger,
                   f"eval finished PARTIALLY — failed suite(s): {failed}. Raw logs are "
                   f"under {run_dir / 'logs'}; fix the cause and re-run the same command "
                   "to resume (completed suites are kept).", 1)
    logger.info("eval complete: %s", run_dir)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PipelineError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
