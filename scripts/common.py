"""Shared utilities for the Keural SFT pipeline.

Every stage script imports from here. Conventions:
- All scripts take --config <stage yaml>; pipeline.yaml is always loaded and merged
  under the "pipeline" key.
- All relative paths in configs resolve against pipeline["project_root"].
- No silent failures: helpers raise PipelineError with actionable messages; stage
  scripts exit non-zero on any unhandled error.
- NEVER initialize CUDA from a coordinator script (another team's job may own the
  GPUs — VERIFIED-FACTS §1). GPU state is read via nvidia-smi subprocess only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

PIPELINE_CONFIG = "configs/pipeline.yaml"

# The venv's REAL storage location — space-free (see venv_python() for why this
# matters). project_root()/venv is a symlink here for interactive convenience
# ("source venv/bin/activate" still works — sourcing a shell script is plain text
# execution, not path-resolving binary invocation, so the symlink there is safe).
VENV_SYMLINK = Path("/home/work/.keural_venv")


class PipelineError(RuntimeError):
    """Fatal, actionable pipeline error."""


# --------------------------------------------------------------------------- config


def project_root() -> Path:
    """Project root = parent of the scripts/ dir containing this file."""
    return Path(__file__).resolve().parent.parent


def subprocess_cwd() -> Path:
    """Working directory for GPU-launching subprocesses (lm_eval, swift sft).

    Not load-bearing for the flashinfer bug (see venv_python()) but kept as
    defense-in-depth: project_root() contains a space and there's no reason for
    any subprocess to depend on CWD, since every path this pipeline builds is
    already absolute.
    """
    d = Path("/home/work/.keural_subprocess_cwd")
    d.mkdir(exist_ok=True)
    return d


def venv_python() -> Path:
    """Python executable to use for EVERY GPU-launching subprocess.

    Root cause (found 2026-07-10): project_root() is "/home/work/keural MoE Pro"
    — contains a space. flashinfer (a vLLM dependency) JIT-compiles CUDA kernels
    on first use and writes -I/-isystem include paths into a generated
    build.ninja file WITHOUT quoting them. Ninja tokenizes on whitespace, so an
    unquoted "-I/home/work/keural MoE Pro/venv/..." splits into "MoE" and
    "Pro/venv/..." as separate tokens and the build fails with "needed by ...:
    missing and no known rule to make it".

    First fix attempt (invoking python through a symlink at a space-free path)
    only partially worked: sys.prefix/sys.executable DID report the symlink's
    path (CPython doesn't resolve symlinks for those), which is why ms-swift's
    subprocess re-exec via sys.executable and the lm_eval crash point both
    appeared fixed initially — but flashinfer itself calls something equivalent
    to os.path.realpath(__file__) when computing include paths, which collapses
    the symlink straight back to the real (space-containing) path. Proved via a
    real end-to-end eval run: it got further before crashing, but crashed later
    at a different flashinfer kernel (verified: freshly-regenerated build.ninja
    still contained the space-containing path after clearing the JIT cache).

    Actual fix: the venv's real storage was physically moved to VENV_SYMLINK
    (space-free), not merely symlinked there — os.path.realpath() cannot resolve
    away a location that's genuinely where the files live. project_root()/venv
    is now a symlink TO this real location (safe for interactive `source
    venv/bin/activate`, unsafe for anything that realpaths its own file).

    Console-script wrappers (venv/bin/lm_eval, venv/bin/swift) still shouldn't
    be relied on for imported-module path resolution — always invoke
    `venv_python()` directly with `-m <module>` (see run_eval.py, train.py).
    """
    if not VENV_SYMLINK.exists():
        raise PipelineError(
            f"{VENV_SYMLINK} is missing — this is where the venv's real files "
            f"must live (project_root() contains a space, which breaks "
            f"flashinfer's CUDA JIT compilation). Recreate the venv there, then "
            f"`ln -s {VENV_SYMLINK} \"{project_root() / 'venv'}\"` for convenience."
        )
    py = VENV_SYMLINK / "bin" / "python"
    if not py.exists():
        raise PipelineError(f"{py} not found — venv install may be broken")
    return py


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise PipelineError(f"Config not found: {p}")
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(stage_config: str | Path | None) -> dict:
    """Load pipeline.yaml, then the stage config on top (stage keys win).

    Returns the stage config dict with the shared config under cfg["pipeline"].
    """
    cfg = load_yaml(stage_config) if stage_config else {}
    cfg["pipeline"] = load_yaml(PIPELINE_CONFIG)
    return cfg


def resolve_path(cfg: dict, rel: str | Path) -> Path:
    root = Path(cfg["pipeline"].get("project_root") or project_root())
    p = Path(rel)
    return p if p.is_absolute() else root / p


def config_fingerprint(obj) -> str:
    """Stable hash of a config fragment — used for idempotency checks."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- logging


def setup_logging(stage: str, cfg: dict | None = None) -> logging.Logger:
    """Console + timestamped file log under logs/<stage>/."""
    log_dir = project_root() / "logs" / stage
    if cfg:
        log_dir = resolve_path(cfg, cfg["pipeline"]["paths"]["logs_dir"]) / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / f"{utc_stamp()}.log"

    logger = logging.getLogger(f"keural.{stage}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("log file: %s", logfile)
    return logger


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- env guards


def guard_environment(logger: logging.Logger | None = None) -> None:
    """Fail-fast checks that this process runs in the intended environment.

    - venv python (not the shared system python)
    - PYTHONPATH must not leak ~/.local (shared with the other team's GPU job)

    Checks against VENV_SYMLINK (the venv's real, space-free location), not
    project_root()/venv — that's a symlink kept only for interactive
    `source venv/bin/activate` convenience; sys.prefix correctly reports the
    real path once activated (see venv_python() for why the real location
    matters at all).
    """
    if VENV_SYMLINK.exists() and not sys.prefix.startswith(str(VENV_SYMLINK)):
        raise PipelineError(
            f"Run with the project venv python ({VENV_SYMLINK}/bin/python), got "
            f"{sys.executable}. Use: source \"{project_root() / 'venv'}/bin/activate\" "
            f"&& export PYTHONPATH="
        )
    leaked = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p.strip()]
    if leaked:
        # Strip rather than die, but loudly: leaking ~/.local shadows the venv stack.
        if logger:
            logger.warning("PYTHONPATH is set (%s) — clearing it for child processes", leaked)
        os.environ["PYTHONPATH"] = ""


def child_env(extra: dict | None = None) -> dict:
    """Environment for subprocesses: venv on PATH, PYTHONPATH cleared.

    Prefers the space-free VENV_SYMLINK/bin over the real venv/bin — see
    venv_python() for why (flashinfer JIT + spaces in project_root()). Falls
    back to the real path if the symlink isn't set up (e.g. a fresh checkout
    that hasn't run env_check.py yet); that fallback is only safe for non-CUDA-
    JIT subprocesses (pip, hf, git, etc.), not for lm_eval/swift — those must
    use venv_python() directly, never rely on PATH lookup.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["PIP_CONSTRAINT"] = ""
    venv_bin = str((VENV_SYMLINK if VENV_SYMLINK.is_symlink() else project_root() / "venv") / "bin")
    if venv_bin not in env.get("PATH", ""):
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def gpu_status() -> list[dict]:
    """GPU list via nvidia-smi (never initializes CUDA in-process)."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0:
        raise PipelineError(f"nvidia-smi failed: {out.stderr.strip()}")
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, used, total, util = [x.strip() for x in line.split(",")]
        # nvidia-smi reports util as "N/A"/"[Not Found]" when the GPU is fully idle
        # at P0 on some driver versions — treat as 0%, not a parse error.
        util_pct = int(util) if util.isdigit() else 0
        gpus.append({"index": int(idx), "mem_used_mib": int(used),
                     "mem_total_mib": int(total), "util_pct": util_pct})
    return gpus


def assert_gpus_free(cfg: dict, logger: logging.Logger) -> None:
    """Refuse to start GPU work while another job occupies the GPUs."""
    threshold = cfg["pipeline"]["runtime"]["gpu_free_mib_threshold"]
    gpus = gpu_status()
    busy = [g for g in gpus if g["mem_used_mib"] > threshold]
    for g in gpus:
        logger.info("GPU%d: %d/%d MiB used, util %d%%",
                    g["index"], g["mem_used_mib"], g["mem_total_mib"], g["util_pct"])
    if busy:
        raise PipelineError(
            f"{len(busy)} GPU(s) busy (>{threshold} MiB used) — another job is running "
            "(do NOT kill it). Re-run when nvidia-smi shows the GPUs free."
        )


def assert_free_disk(cfg: dict, stage: str, logger: logging.Logger) -> None:
    need_gb = cfg["pipeline"]["runtime"]["min_free_disk_gb"].get(stage)
    if not need_gb:
        return
    free_gb = shutil.disk_usage(project_root()).free / 1e9
    logger.info("free disk: %.0f GB (need %d GB for stage %r)", free_gb, need_gb, stage)
    if free_gb < need_gb:
        raise PipelineError(
            f"Only {free_gb:.0f} GB free on {project_root()}, stage {stage!r} needs "
            f"{need_gb} GB. Clear space (old checkpoints/outputs?) and retry."
        )


# --------------------------------------------------------------------------- files


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_json_atomic(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_logged(cmd: list[str], logger: logging.Logger, log_path: str | Path,
               env: dict | None = None, cwd: str | Path | None = None) -> int:
    """Run a subprocess, teeing stdout+stderr to a file and echoing to the logger.

    Returns the exit code (caller decides whether to raise).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("exec: %s", " ".join(cmd))
    logger.info("  (full output -> %s)", log_path)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {utc_stamp()} exec: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env or child_env(), cwd=str(cwd or subprocess_cwd()),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            line = line.rstrip()
            if line:
                logger.debug(line)
        proc.wait()
    if proc.returncode != 0:
        logger.error("command failed (exit %d): %s", proc.returncode, " ".join(cmd))
    return proc.returncode


# --------------------------------------------------------------------------- misc


def die(logger: logging.Logger, msg: str, code: int = 1) -> None:
    logger.error(msg)
    sys.exit(code)


def stage_banner(logger: logging.Logger, name: str, cfg_path: str | None) -> None:
    logger.info("=" * 70)
    logger.info("Keural SFT pipeline — stage: %s", name)
    logger.info("config: %s | started: %s", cfg_path, datetime.now().isoformat())
    logger.info("=" * 70)
