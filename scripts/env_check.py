#!/usr/bin/env python
"""Environment check for the Keural SFT pipeline.

Two modes:

* default (CPU-safe, run anytime): venv/python correctness, pinned package
  versions (VERIFIED-FACTS §1), PYTHONPATH/PIP_CONSTRAINT hazards, per-stage
  disk thresholds, GPU status (informational, via nvidia-smi — never CUDA),
  HF token, base-model config + dataset repos reachable at their PINNED
  revisions, local keural-SFT cache, fast-kernel importability
  (flash-linear-attention / causal-conv1d), lm_eval task registry.

* --gpu (run ONLY when nvidia-smi shows the GPUs free): refuses via
  common.assert_gpus_free first, then — in SUBPROCESSES, never in-process —
  CUDA init + matmul on each GPU, a 2-process NCCL all-reduce via torchrun,
  vllm import, and a torch-CUDA-version vs driver sanity check (this driver is
  570.x => torch must be cu12x; a cu13 wheel means the venv was re-pinned
  wrongly and will not run — VERIFIED-FACTS §1).

WHY subprocesses for anything CUDA: another team's job may own the GPUs; even
`torch.cuda.is_available()` in the coordinator process is forbidden
(VERIFIED-FACTS §1).  Subprocesses also isolate hard crashes (broken .so ABI)
so this checker always finishes and prints its table.

Exit code: non-zero iff any check FAILs (WARNs don't fail the run).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import common
from common import PipelineError

# Pinned stack, VERIFIED-FACTS §1 (installed 2026-07-08).  Missing = FAIL,
# version mismatch = WARN (a deliberate re-pin should update this table).
PINNED_VERSIONS = {
    "torch": "2.11.0+cu129",
    "torchvision": "0.26.0+cu129",
    "torchaudio": "2.11.0+cu129",
    "vllm": "0.24.0+cu129",
    "transformers": "5.12.1",
    "ms-swift": "4.4.0",
    "peft": "0.19.1",
    "trl": "0.29.1",
    "accelerate": "1.14.0",
    "deepspeed": "0.19.2",
    "lm_eval": "0.4.12",
    "datasets": "4.8.4",
    "qwen-vl-utils": None,   # presence-only (ms-swift requirement for this model)
    "decord": None,          # presence-only
    "huggingface_hub": None,  # presence-only (CLI surface checked by publish.py)
}

REQUIRED_LM_EVAL_TASKS = ["kmmlu", "haerae", "kobest", "mmlu", "gsm8k", "humaneval"]

NET_TIMEOUT_S = 20


class Results:
    """Collect (status, name, detail) rows; render a summary table at the end."""

    ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2, "INFO": 3}

    def __init__(self, logger):
        self.rows: list[tuple[str, str, str]] = []
        self.logger = logger

    def add(self, status: str, name: str, detail: str) -> None:
        self.rows.append((status, name, detail))
        log = {"FAIL": self.logger.error, "WARN": self.logger.warning}.get(
            status, self.logger.info)
        log("[%s] %s: %s", status, name, detail)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.rows)

    def summary(self) -> str:
        width = max(len(n) for _, n, _ in self.rows)
        lines = ["", "=" * 78, "ENVIRONMENT CHECK SUMMARY", "=" * 78]
        for status, name, detail in self.rows:
            lines.append(f"{status:<5} {name:<{width}}  {detail}")
        counts = {}
        for status, _, _ in self.rows:
            counts[status] = counts.get(status, 0) + 1
        lines.append("-" * 78)
        lines.append("  ".join(f"{k}={v}" for k, v in sorted(counts.items(),
                                                             key=lambda kv: self.ORDER[kv[0]])))
        return "\n".join(lines)


def run_py(code: str, timeout: int, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a python snippet in a clean venv subprocess (PYTHONPATH cleared)."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=timeout, env=common.child_env(extra_env))


def tail(proc: subprocess.CompletedProcess, n: int = 3) -> str:
    text = (proc.stderr or proc.stdout or "").strip()
    return " | ".join(text.splitlines()[-n:]) if text else "<no output>"


# ------------------------------------------------------------- default checks


def check_python_env(res: Results) -> None:
    try:
        common.guard_environment(res.logger)
        res.add("PASS", "venv", f"python {sys.version.split()[0]} at {sys.prefix}")
    except PipelineError as exc:
        res.add("FAIL", "venv", str(exc))


def check_env_hazards(res: Results) -> None:
    # guard_environment() clears PYTHONPATH in-process, so inspect the parent
    # shell's values instead of our own (already-sanitized) environment.
    proc = subprocess.run(["bash", "-lc", "echo \"$PYTHONPATH\"; echo \"$PIP_CONSTRAINT\""],
                          capture_output=True, text=True, timeout=30)
    login_pythonpath, login_pip = (proc.stdout.splitlines() + ["", ""])[:2]
    if login_pythonpath.strip():
        res.add("WARN", "PYTHONPATH",
                f"login shells export PYTHONPATH={login_pythonpath!r} (shadows the venv "
                "with the shared ~/.local stack) — always `export PYTHONPATH=` first")
    else:
        res.add("PASS", "PYTHONPATH", "not exported by login shells")
    if login_pip.strip():
        res.add("WARN", "PIP_CONSTRAINT",
                f"login shells export PIP_CONSTRAINT={login_pip!r} (NVIDIA pins) — "
                "any venv pip operation must run with PIP_CONSTRAINT= (empty)")
    else:
        res.add("PASS", "PIP_CONSTRAINT", "not exported by login shells")


def check_versions(res: Results) -> None:
    from importlib.metadata import PackageNotFoundError, version

    for dist, pinned in PINNED_VERSIONS.items():
        try:
            found = version(dist)
        except PackageNotFoundError:
            res.add("FAIL", f"pkg:{dist}",
                    "MISSING from the venv — reinstall per VERIFIED-FACTS §1 "
                    "(with PIP_CONSTRAINT= and the venv active)")
            continue
        if pinned is None or found == pinned:
            res.add("PASS", f"pkg:{dist}", found)
        else:
            res.add("WARN", f"pkg:{dist}",
                    f"{found} != pinned {pinned} (re-pin was deliberate? then update "
                    "PINNED_VERSIONS in env_check.py)")


def check_disk(res: Results, cfg: dict) -> None:
    free_gb = shutil.disk_usage(common.project_root()).free / 1e9
    thresholds = cfg["pipeline"]["runtime"]["min_free_disk_gb"]
    if free_gb < 20:
        res.add("FAIL", "disk", f"only {free_gb:.0f} GB free — below any usable margin")
        return
    for stage, need in sorted(thresholds.items()):
        if free_gb < need:
            res.add("WARN", f"disk:{stage}",
                    f"{free_gb:.0f} GB free < {need} GB required — stage {stage!r} "
                    "will refuse to start until space is cleared")
        else:
            res.add("PASS", f"disk:{stage}", f"{free_gb:.0f} GB free >= {need} GB")


def check_gpus_informational(res: Results, cfg: dict) -> None:
    try:
        gpus = common.gpu_status()
    except PipelineError as exc:
        res.add("FAIL", "nvidia-smi", str(exc))
        return
    threshold = cfg["pipeline"]["runtime"]["gpu_free_mib_threshold"]
    busy = [g for g in gpus if g["mem_used_mib"] > threshold]
    detail = "; ".join(f"GPU{g['index']} {g['mem_used_mib']}/{g['mem_total_mib']} MiB "
                       f"util {g['util_pct']}%" for g in gpus)
    res.add("INFO", "gpus", detail + (f" — {len(busy)} BUSY (do not start GPU work)"
                                      if busy else " — all free"))


def check_hf_token(res: Results) -> None:
    from huggingface_hub import get_token

    if get_token():
        res.add("PASS", "hf-token", "token present (huggingface_hub.get_token)")
    else:
        res.add("WARN", "hf-token",
                "no HF token — private repos and publishing will fail; run "
                "venv/bin/hf auth login")


def check_model_config(res: Results, cfg: dict) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    model_id = cfg["pipeline"]["model"]["id"]
    revision = cfg["pipeline"]["model"]["revision"]
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(NET_TIMEOUT_S))
    try:
        path = hf_hub_download(model_id, "config.json", revision=revision)
        res.add("PASS", "model-config", f"{model_id}@{revision[:12]} -> {path}")
    except LocalEntryNotFoundError:
        res.add("FAIL", "model-config",
                f"{model_id}@{revision[:12]} unreachable and not cached — check "
                "network/token/revision pin")
    except Exception as exc:
        try:
            path = hf_hub_download(model_id, "config.json", revision=revision,
                                   local_files_only=True)
            res.add("WARN", "model-config",
                    f"hub unreachable ({type(exc).__name__}) but pinned config is "
                    f"cached at {path}")
        except Exception:
            res.add("FAIL", "model-config",
                    f"{model_id}@{revision[:12]}: {type(exc).__name__}: {exc}")


def check_dataset_repos(res: Results) -> None:
    from huggingface_hub import HfApi

    try:
        sources = common.load_yaml("configs/dataset.yaml").get("sources", [])
    except PipelineError as exc:
        res.add("FAIL", "dataset-config", str(exc))
        return
    api = HfApi()
    seen = set()
    for src in sources:
        repo, revision = src.get("repo"), src.get("revision")
        if not repo or (repo, revision) in seen:
            continue
        seen.add((repo, revision))
        try:
            api.dataset_info(repo, revision=revision, timeout=NET_TIMEOUT_S)
            res.add("PASS", f"dataset:{repo.split('/')[-1]}", f"@{str(revision)[:12]} reachable")
        except Exception as exc:
            res.add("FAIL", f"dataset:{repo.split('/')[-1]}",
                    f"@{str(revision)[:12]}: {type(exc).__name__}: {exc}")


def check_keural_sft_cache(res: Results) -> None:
    """keural-SFT is 115 shards; VERIFIED-FACTS §4 says it is fully in the local
    HF cache — absence means the dataset stage will re-download ~everything."""
    from huggingface_hub.constants import HF_HUB_CACHE

    pinned = None
    for src in common.load_yaml("configs/dataset.yaml").get("sources", []):
        if src.get("repo") == "mkd-chanwoo/keural-SFT":
            pinned = src.get("revision")
            break
    cache_dir = Path(HF_HUB_CACHE) / "datasets--mkd-chanwoo--keural-SFT"
    if not cache_dir.exists():
        res.add("WARN", "keural-SFT-cache",
                f"{cache_dir} missing — dataset stage will download 115 shards")
        return
    snapshots = [p.name for p in (cache_dir / "snapshots").glob("*")] \
        if (cache_dir / "snapshots").exists() else []
    if pinned and pinned in snapshots:
        n_files = sum(1 for _ in (cache_dir / "snapshots" / pinned).rglob("*") if _.is_file())
        res.add("PASS", "keural-SFT-cache", f"pinned snapshot cached ({n_files} files)")
    else:
        res.add("WARN", "keural-SFT-cache",
                f"cache exists but pinned revision {str(pinned)[:12]} not among "
                f"snapshots {snapshots} — expect (partial) re-download")


def check_fast_kernels(res: Results) -> None:
    """flash-linear-attention + causal-conv1d power the linear-attention fast
    path in 30/40 layers.  Broken/missing => slow torch fallback (WARN)."""
    for module, dist in (("fla", "flash-linear-attention"),
                         ("causal_conv1d", "causal-conv1d")):
        proc = run_py(f"import {module}; print(getattr({module}, '__version__', 'ok'))",
                      timeout=120)
        if proc.returncode == 0:
            res.add("PASS", f"kernel:{dist}", proc.stdout.strip() or "importable")
        else:
            res.add("WARN", f"kernel:{dist}",
                    f"import failed ({tail(proc, 1)}) — transformers will fall back "
                    "to the SLOW torch linear-attention path on 30/40 layers")


def check_lm_eval_tasks(res: Results) -> None:
    code = (
        "from lm_eval.tasks import TaskManager\n"
        "names = set(TaskManager().all_tasks)\n"
        f"required = {REQUIRED_LM_EVAL_TASKS!r}\n"
        "missing = [t for t in required if t not in names]\n"
        "print('MISSING=' + ','.join(missing))\n"
    )
    try:
        proc = run_py(code, timeout=300)
    except subprocess.TimeoutExpired:
        res.add("FAIL", "lm_eval-tasks", "task registry scan timed out (300s)")
        return
    if proc.returncode != 0:
        res.add("FAIL", "lm_eval-tasks", f"registry scan crashed: {tail(proc)}")
        return
    missing_line = [ln for ln in proc.stdout.splitlines() if ln.startswith("MISSING=")]
    missing = missing_line[0].split("=", 1)[1] if missing_line else "?"
    if missing:
        res.add("FAIL", "lm_eval-tasks",
                f"required tasks missing from registry: {missing}")
    else:
        res.add("PASS", "lm_eval-tasks", "kmmlu haerae kobest mmlu gsm8k humaneval all present")


# ----------------------------------------------------------------- gpu checks

CUDA_WORKER = """\
import sys, torch
dev = int(sys.argv[1]) if len(sys.argv) > 1 else 0
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
torch.cuda.set_device(dev)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = (a @ a).float().abs().mean().item()
torch.cuda.synchronize()
print(f"OK gpu={dev} name={torch.cuda.get_device_name(dev)} "
      f"torch_cuda={torch.version.cuda} matmul_mean={c:.3f}")
"""

NCCL_WORKER = """\
import os, torch, torch.distributed as dist
dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)
t = torch.ones(1024, device="cuda")
dist.all_reduce(t)
assert t[0].item() == dist.get_world_size(), f"all_reduce got {t[0].item()}"
print(f"OK rank={rank}/{dist.get_world_size()} all_reduce verified")
dist.destroy_process_group()
"""


def gpu_checks(res: Results, cfg: dict) -> None:
    logger = res.logger
    # Hard gate: NEVER touch GPUs that another job occupies.
    common.assert_gpus_free(cfg, logger)
    res.add("PASS", "gpus-free", "no other job holds the GPUs")

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
    driver_version = driver[0] if driver else "?"
    res.add("INFO", "driver", f"nvidia driver {driver_version}")

    with tempfile.TemporaryDirectory(prefix="keural-envcheck-") as tmp:
        worker = Path(tmp) / "cuda_worker.py"
        worker.write_text(CUDA_WORKER, encoding="utf-8")
        n_gpus = len(common.gpu_status())
        torch_cuda = None
        for idx in range(n_gpus):
            proc = subprocess.run([sys.executable, str(worker), str(idx)],
                                  capture_output=True, text=True, timeout=300,
                                  env=common.child_env())
            if proc.returncode == 0:
                out = proc.stdout.strip()
                res.add("PASS", f"cuda:gpu{idx}", out)
                for token in out.split():
                    if token.startswith("torch_cuda="):
                        torch_cuda = token.split("=", 1)[1]
            else:
                res.add("FAIL", f"cuda:gpu{idx}", f"init/matmul failed: {tail(proc)}")

        if torch_cuda is not None:
            if torch_cuda.startswith("12."):
                res.add("PASS", "cuda-version",
                        f"torch built for CUDA {torch_cuda}, driver {driver_version} (12.x OK)")
            else:
                res.add("FAIL", "cuda-version",
                        f"torch built for CUDA {torch_cuda} but driver {driver_version} "
                        "(570.x) only runs CUDA 12.x — the venv was re-pinned to a cu13 "
                        "wheel; reinstall torch 2.11.0+cu129 (VERIFIED-FACTS §1)")

        nccl = Path(tmp) / "nccl_worker.py"
        nccl.write_text(NCCL_WORKER, encoding="utf-8")
        # venv_python() -m torch.distributed.run, not venv/bin/torchrun — that
        # console-script wrapper's shebang hardcodes the real space-containing
        # venv path, which breaks any CUDA-JIT-compiling dependency (flashinfer,
        # possibly torch's own cpp_extension). See common.venv_python().
        proc = subprocess.run(
            [str(common.venv_python()), "-m", "torch.distributed.run",
             "--nproc_per_node=2", "--master_port=29517", str(nccl)],
            capture_output=True, text=True, timeout=600,
            env=common.child_env({"CUDA_VISIBLE_DEVICES": "0,1"}))
        if proc.returncode == 0 and proc.stdout.count("OK rank=") == 2:
            res.add("PASS", "nccl", "torchrun 2-process all_reduce verified")
        else:
            res.add("FAIL", "nccl", f"torchrun all_reduce failed: {tail(proc, 5)}")

    proc = run_py("import vllm; print(vllm.__version__)", timeout=300)
    if proc.returncode == 0:
        res.add("PASS", "vllm-import", proc.stdout.strip())
    else:
        res.add("FAIL", "vllm-import", f"import failed: {tail(proc)}")


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=None,
                        help="optional stage yaml; pipeline.yaml is always loaded "
                             "(env_check has no stage config of its own)")
    parser.add_argument("--gpu", action="store_true",
                        help="ALSO run GPU runtime checks (only when GPUs are free!)")
    parser.add_argument("--skip-network", action="store_true",
                        help="skip hub reachability checks (offline hosts)")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("env_check", cfg)
    common.stage_banner(logger, "env_check" + (" --gpu" if args.gpu else ""), args.config)
    res = Results(logger)

    try:
        check_python_env(res)
        check_env_hazards(res)
        check_versions(res)
        check_disk(res, cfg)
        check_gpus_informational(res, cfg)
        check_hf_token(res)
        if args.skip_network:
            res.add("INFO", "network-checks", "skipped (--skip-network)")
        else:
            check_model_config(res, cfg)
            check_dataset_repos(res)
        check_keural_sft_cache(res)
        check_fast_kernels(res)
        check_lm_eval_tasks(res)
        if args.gpu:
            gpu_checks(res, cfg)
    except PipelineError as exc:
        # e.g. assert_gpus_free refusing while the other team's job runs
        res.add("FAIL", "aborted", str(exc))

    print(res.summary())
    if res.failed:
        logger.error("environment check FAILED — fix the FAIL rows above")
        sys.exit(1)
    logger.info("environment check passed (no FAILs)")


if __name__ == "__main__":
    main()
