#!/usr/bin/env python
"""Stage 7c — upload the merged Keural model to the Hugging Face Hub.

Uses the CURRENT `hf` CLI (huggingface_hub 1.22; `huggingface-cli` is
deprecated).  Subcommands verified against the installed venv/bin/hf on
2026-07-08:

* ``hf auth whoami``          — token check
* ``hf repo create <id> --private/--public --exist-ok``
* ``hf upload <id> <folder> .`` — recommended single-commit path
* ``hf upload-large-folder``  — still present but marked "[Deprecated] Use
  `hf upload` instead" in 1.22; kept available behind --large for
  restart-resumable multi-commit uploads of the ~70 GB folder.

Refuses to publish unless the merged dir carries a passing merge_report.json
(produced only after merge_lora.py's byte-level verification) AND a README.md
model card — an unverified or undocumented 70 GB upload is exactly the kind of
silent failure this pipeline exists to prevent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import common
from common import PipelineError

DEFAULT_CONFIG = "configs/publish.yaml"


def hf_cli() -> str:
    """Absolute path to the venv's hf CLI — never whatever is first on PATH."""
    exe = common.project_root() / "venv" / "bin" / "hf"
    if not exe.exists():
        raise PipelineError(f"{exe} not found — the project venv is incomplete "
                            "(pip show huggingface_hub inside the venv).")
    return str(exe)


def whoami(logger) -> str:
    """Return the logged-in HF username, or raise with an actionable message."""
    proc = subprocess.run([hf_cli(), "auth", "whoami"], capture_output=True,
                          text=True, env=common.child_env(), timeout=60)
    if proc.returncode != 0:
        raise PipelineError(
            "No valid Hugging Face token (`hf auth whoami` failed: "
            f"{(proc.stderr or proc.stdout).strip().splitlines()[-1:]}). Fix: run\n"
            f'  "{hf_cli()}" auth login\n'
            "with a WRITE-scoped token for the target org/user."
        )
    out = proc.stdout.strip()
    logger.info("hf auth whoami: %s", out.replace("\n", " | "))
    return out.replace("\n", " | ") if out else "?"


def check_namespace(user_line: str, repo: str, logger) -> None:
    """WARN loudly when the token's identity can't obviously write to the target
    namespace. Known live issue on this box: token user is `mkd-hika` (orgs:
    mkd-ai) while the publish target is `mkd-hossain/...` — `whoami` passes but
    `hf repo create` would 403. Warning rather than failing because org
    membership / fine-grained tokens can grant access we cannot see here."""
    namespace = repo.split("/")[0]
    if namespace.lower() not in user_line.lower():
        logger.warning(
            "HF token identity (%s) does not mention target namespace %r — if the "
            "token cannot write to %r, `hf repo create` will fail with 403. Fix: "
            "hf auth login with a WRITE token for %r.",
            user_line, namespace, repo, namespace,
        )


def preflight(merged_dir: Path, logger) -> dict:
    """Refuse to upload anything that is not a verified, documented merge."""
    report_path = merged_dir / "merge_report.json"
    if not report_path.exists():
        raise PipelineError(
            f"{report_path} missing — merged dir is unverified (run "
            "scripts/merge_lora.py first; a merge that skipped verification must "
            "never be published).")
    report = common.read_json(report_path)
    if report.get("status") != "ok":
        raise PipelineError(f"merge_report.json status is {report.get('status')!r}, "
                            "not 'ok' — refusing to publish.")
    if not (merged_dir / "README.md").exists():
        raise PipelineError(f"{merged_dir}/README.md missing — generate the model card "
                            "first (scripts/make_model_card.py).")
    for required in ("config.json", "model.safetensors.index.json"):
        if not (merged_dir / required).exists():
            raise PipelineError(f"{merged_dir}/{required} missing — not a complete "
                                "merged model dir.")
    files = sorted(p for p in merged_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    logger.info("preflight OK: %d files, %.1f GB, merge fingerprint %s",
                len(files), total / 1e9, report.get("fingerprint"))
    return {"n_files": len(files), "total_bytes": total,
            "fingerprint": report.get("fingerprint")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--merged-dir", default=None,
                        help="override cfg merge.merged_dir")
    parser.add_argument("--repo", default=None, help="override cfg upload.repo")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan; touch nothing on the Hub")
    parser.add_argument("--large", action="store_true",
                        help="use `hf upload-large-folder` (multi-commit, resumable "
                             "across restarts) instead of single-commit `hf upload`")
    args = parser.parse_args()

    cfg = common.load_config(args.config)
    logger = common.setup_logging("publish", cfg)
    common.stage_banner(logger, "publish", args.config)
    common.guard_environment(logger)

    try:
        ucfg = cfg.get("upload") or {}
        repo = args.repo or ucfg.get("repo") or cfg["pipeline"]["hf"]["publish_repo"]
        private = bool(ucfg.get("private", cfg["pipeline"]["hf"].get("private", True)))
        # Always resolve to absolute: the hf subprocess runs from
        # common.subprocess_cwd() (space-free dir), NOT the project root, so a
        # relative path that works for our own preflight checks would 404 there
        # (bit us on 2026-07-16: "No such file or directory" from hf upload).
        merged_dir = common.resolve_path(
            cfg, args.merged_dir if args.merged_dir
            else (cfg.get("merge") or {}).get("merged_dir", "outputs/merged"))
        if not merged_dir.is_dir():
            raise PipelineError(f"merged dir {merged_dir} does not exist — run "
                                "scripts/merge_lora.py first.")

        stats = preflight(merged_dir, logger)

        create_cmd = [hf_cli(), "repo", "create", repo,
                      "--private" if private else "--public", "--exist-ok"]
        if args.large:
            upload_cmd = [hf_cli(), "upload-large-folder", repo, str(merged_dir),
                          "--repo-type", "model"]
        else:
            upload_cmd = [hf_cli(), "upload", repo, str(merged_dir), ".",
                          "--repo-type", "model", "--commit-message",
                          f"Publish Keural merged model (merge {stats['fingerprint']})"]
        url = f"https://huggingface.co/{repo}"

        if args.dry_run:
            try:
                user = whoami(logger)
                logger.info("[dry-run] token OK (%s)", user)
                check_namespace(user, repo, logger)
            except PipelineError as exc:
                logger.warning("[dry-run] token check failed (would block a real "
                               "publish): %s", exc)
            logger.info("[dry-run] plan:")
            logger.info("[dry-run]   source : %s (%d files, %.1f GB)",
                        merged_dir, stats["n_files"], stats["total_bytes"] / 1e9)
            logger.info("[dry-run]   target : %s (%s)", url,
                        "private" if private else "public")
            logger.info("[dry-run]   step 1 : %s", " ".join(create_cmd))
            logger.info("[dry-run]   step 2 : %s", " ".join(upload_cmd))
            logger.info("[dry-run] nothing was created or uploaded.")
            return

        user = whoami(logger)
        check_namespace(user, repo, logger)
        log_path = common.project_root() / "logs" / "publish" / f"hf-{common.utc_stamp()}.log"
        if common.run_logged(create_cmd, logger, log_path) != 0:
            raise PipelineError(f"`hf repo create` failed — see {log_path}. Common "
                                "causes: token lacks write scope for "
                                f"{repo.split('/')[0]!r}, or org permissions.")
        logger.info("uploading %s -> %s (%.1f GB)...", merged_dir, repo,
                    stats["total_bytes"] / 1e9)
        if common.run_logged(upload_cmd, logger, log_path) != 0:
            raise PipelineError(
                f"upload failed — see {log_path}. The upload is content-addressed; "
                "re-running this script resumes/skips already-uploaded files "
                "(or retry with --large for the multi-commit resumable path).")
        logger.info("PUBLISH OK -> %s", url)
        print(url)
    except PipelineError as exc:
        common.die(logger, str(exc))


if __name__ == "__main__":
    main()
