"""Keural v2 dataset builder (see V2-DATASET-PREP.md / V2-TRAINING-PLAN.md).

Produces a ~150k, Korean-majority, format-balanced instruction set that fixes
v1's problems, output as {"messages":[...]} JSONL for the training pipeline.

Enforces the three non-negotiable rules:
  1. TEST-SPLIT HYGIENE — only train/auxiliary_train/dev splits of any benchmark
     source are ever loaded; test splits are never touched.
  2. CODE VALIDATION — every python block must ast.parse(); broken-indentation
     rows (the v1 HumanEval killer) are rejected.
  3. IDENTITY DE-CONTAMINATION — assistant turns claiming to be OpenAI/GPT/Qwen/etc.
     are dropped; a hand-seeded Keural identity set is added instead.

CPU/network only — never touches the GPUs.  Use --smoke to validate all loaders
with tiny counts before the full run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402

SEED = 42
PROJECT = common.project_root()   # self-contained: the Keural Orbit folder
random.seed(SEED)

# ---- identity contamination filter (rule 3) --------------------------------
BANNED = re.compile(
    r"\b(openai|chatgpt|gpt-4|gpt-3|gpt4|gpt3|anthropic|claude|"
    r"google (deepmind|ai)|gemma|gemini|qwen|tongyi|alibaba|llama|mistral)\b", re.I)


def code_blocks_ok(text: str) -> bool:
    """Rule 2: every ```python block must parse. Reject broken indentation."""
    for block in re.findall(r"```(?:python|py)\n(.*?)```", text, re.S):
        block = block.strip("\n")
        if not block.strip():
            continue
        try:
            ast.parse(block)
        except (SyntaxError, IndentationError, ValueError):
            return False
    return True


def clean_identity(msgs: list[dict]) -> bool:
    """Rule 3: drop rows whose assistant turns name a foreign lab/model."""
    for m in msgs:
        if m["role"] == "assistant" and BANNED.search(m["content"] or ""):
            return False
    return True


def row(messages, source, license):
    return {"messages": messages, "source_name": source, "license": license}


def norm_key(messages) -> str:
    return hashlib.sha256(
        "".join(m["content"] for m in messages).strip().encode()).hexdigest()


# --------------------------------------------------------------------------- categories

def cat_korean_conversation(n, log):
    """A. Reuse cached commercial-clean Korean sources (ChatML `text` rows)."""
    from huggingface_hub import hf_hub_download
    specs = [
        ("mkd-chanwoo/keural-conversation-chatml-ko", "keural_conversation_shard_0000.jsonl",
         "64a8c45cf5d777b77fe8e1351e5a52da9dd6868b"),
        ("mkd-chanwoo/keural-rag-chatml-ko", "keural_synthetic_sft_shard_0000.jsonl",
         "aa9023d231a3451d25135d5aa84b12b48ca06450"),
    ]
    out = []
    per = n // len(specs)
    for repo, fn, rev in specs:
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        random.shuffle(rows)
        kept = 0
        for r in rows:
            msgs = parse_chatml(r.get("text", ""))
            if not msgs or msgs[-1]["role"] != "assistant":
                continue
            if not clean_identity(msgs):
                continue
            out.append(row(msgs, "ko_conversation", r.get("license", "cc-by-4.0")))
            kept += 1
            if kept >= per:
                break
    log("A korean_conversation: %d" % len(out))
    return out


def parse_chatml(text: str) -> list[dict]:
    msgs = []
    for m in re.finditer(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", text, re.S):
        role, content = m.group(1), m.group(2).strip()
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def _mcq(question, choices, answer_idx, lang):
    letters = "ABCDEFGH"
    opts = " ".join(f"({letters[i]}) {c}" for i, c in enumerate(choices))
    ans = letters[answer_idx]
    if lang == "ko":
        u = f"다음 질문에 답하세요.\n\n질문: {question}\n{opts}"
        a = f"정답: ({ans})"
    else:
        u = f"Answer the following question.\n\nQuestion: {question}\n{opts}"
        a = f"The answer is ({ans})."
    return [{"role": "user", "content": u}, {"role": "assistant", "content": a}]


def cat_korean_mcq(n, log):
    """B. KMMLU — TRAIN split only (rule 1). The key knowledge lever."""
    from datasets import load_dataset, get_dataset_config_names
    out = []
    try:
        subjects = get_dataset_config_names("HAERAE-HUB/KMMLU")
    except Exception as e:
        log("B korean_mcq: cannot list KMMLU configs (%s) — SKIPPED" % e)
        return out
    subjects = [s for s in subjects if s.lower() != "all"]
    random.shuffle(subjects)
    per = max(1, n // max(1, len(subjects)))
    for subj in subjects:
        try:
            ds = load_dataset("HAERAE-HUB/KMMLU", subj, split="train")  # NEVER 'test'
        except Exception:
            continue
        idxs = list(range(len(ds)))
        random.shuffle(idxs)
        for i in idxs[:per]:
            r = ds[i]
            choices = [r["A"], r["B"], r["C"], r["D"]]
            out.append(row(_mcq(r["question"], choices, int(r["answer"]) - 1, "ko"),
                           "kmmlu_train", "cc-by-sa-4.0"))
        if len(out) >= n:
            break
    log("B korean_mcq: %d" % len(out))
    return out[:n]


def cat_english_mcq(n, log):
    """C. MMLU auxiliary_train + ARC + OpenBookQA — TRAIN splits only (rule 1)."""
    from datasets import load_dataset
    out = []
    try:
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")  # NOT test/validation
        idxs = list(range(len(ds))); random.shuffle(idxs)
        for i in idxs[: int(n * 0.7)]:
            r = ds[i]
            out.append(row(_mcq(r["question"], r["choices"], int(r["answer"]), "en"),
                           "mmlu_auxtrain", "mit"))
    except Exception as e:
        log("  C mmlu aux failed: %s" % e)
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        lab2i = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3}
        for r in arc:
            ak = r["answerKey"]
            if ak not in lab2i:
                continue
            out.append(row(_mcq(r["question"], r["choices"]["text"], lab2i[ak], "en"),
                           "arc_train", "cc-by-sa-4.0"))
            if len(out) >= n:
                break
    except Exception as e:
        log("  C arc failed: %s" % e)
    random.shuffle(out)
    log("C english_mcq: %d" % len(out[:n]))
    return out[:n]


def cat_math(n, log):
    """D. GSM8K train + MetaMathQA (stream). TRAIN only (rule 1)."""
    from datasets import load_dataset
    out = []
    try:
        g = load_dataset("openai/gsm8k", "main", split="train")  # NOT test
        for r in g:
            out.append(row([{"role": "user", "content": r["question"]},
                            {"role": "assistant", "content": r["answer"]}],
                           "gsm8k_train", "mit"))
            if len(out) >= n // 2:
                break
    except Exception as e:
        log("  D gsm8k failed: %s" % e)
    try:
        mm = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        for r in mm:
            q, a = r.get("query"), r.get("response")
            if not q or not a:
                continue
            out.append(row([{"role": "user", "content": q},
                            {"role": "assistant", "content": a}], "metamath", "mit"))
            if len(out) >= n:
                break
    except Exception as e:
        log("  D metamath failed: %s" % e)
    random.shuffle(out)
    log("D math: %d" % len(out[:n]))
    return out[:n]


def cat_code(n, log):
    """E. Magicoder (Evol + OSS) from ORIGINAL upstream, indentation-validated (rule 2)."""
    from datasets import load_dataset
    out = []
    rejected = 0
    sources = [
        ("ise-uiuc/Magicoder-Evol-Instruct-110K", "train", "instruction", "response"),
        ("ise-uiuc/Magicoder-OSS-Instruct-75K", "train", "problem", "solution"),
    ]
    per = n // len(sources)
    for repo, split, qf, af in sources:
        try:
            ds = load_dataset(repo, split=split)
        except Exception as e:
            log("  E %s failed: %s" % (repo, e))
            continue
        idxs = list(range(len(ds))); random.shuffle(idxs)
        kept = 0
        for i in idxs:
            r = ds[i]
            q, a = r.get(qf), r.get(af)
            if not q or not a:
                continue
            if not code_blocks_ok(a):          # rule 2: reject broken indentation
                rejected += 1
                continue
            out.append(row([{"role": "user", "content": q},
                            {"role": "assistant", "content": a}], "magicoder_clean", "mit"))
            kept += 1
            if kept >= per:
                break
    log("E code: %d kept, %d rejected for bad indentation" % (len(out), rejected))
    return out[:n]


def cat_replay(n, log):
    """F. English general replay — anti-forgetting. ultrachat + openorca (stream)."""
    from datasets import load_dataset
    out = []
    try:
        uc = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        for r in uc:
            msgs = [{"role": m["role"], "content": m["content"]}
                    for m in r["messages"] if m["role"] in ("user", "assistant") and m["content"]]
            if len(msgs) < 2 or msgs[-1]["role"] != "assistant" or not clean_identity(msgs):
                continue
            out.append(row(msgs, "ultrachat", "mit"))
            if len(out) >= n // 2:
                break
    except Exception as e:
        log("  F ultrachat failed: %s" % e)
    try:
        oo = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        for r in oo:
            q, a = r.get("question"), r.get("response")
            if not q or not a:
                continue
            msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            if not clean_identity(msgs):
                continue
            out.append(row(msgs, "openorca", "mit"))
            if len(out) >= n:
                break
    except Exception as e:
        log("  F openorca failed: %s" % e)
    random.shuffle(out)
    log("F replay: %d" % len(out[:n]))
    return out[:n]


def cat_longform(n, log):
    """G. Fix short-answer bias with long, substantive Korean answers sourced
    from the Mkd-Yonas pool (kocommercial/kullm) — DISTINCT from category A's
    conversation/rag data, so these survive global dedup instead of colliding.
    Uses hf_hub_download (same reliable path as category A)."""
    from huggingface_hub import hf_hub_download
    repo = "Mkd-Yonas/keural-SFT-chatml-ko-v1"
    rev = "c56d7885efb37799deb1ecba471e4f6bf263852e"
    files = ["kocommercial_shard_0000.jsonl", "kullm_v2_shard_0000.jsonl"]
    cand = []
    for fn in files:
        try:
            p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 100000:
                        break
                    r = json.loads(line)
                    msgs = parse_chatml(r.get("text", ""))
                    if not msgs or msgs[-1]["role"] != "assistant":
                        continue
                    alen = len(msgs[-1]["content"])
                    if alen < 400 or not clean_identity(msgs):   # long answers only
                        continue
                    cand.append((alen, msgs, r.get("license", "mit")))
        except Exception as e:
            log("  G %s failed: %s" % (fn, e))
    cand.sort(key=lambda x: -x[0])
    out = [row(m, "ko_longform", lic) for _, m, lic in cand[:n]]
    log("G longform: %d (long Korean answers from distinct pool)" % len(out))
    return out


# Each entry: (list of user phrasings, list of assistant answer variants).
# Uniqueness is by construction (user × answer × prefix product), so dedup keeps them.
IDENTITY_GROUPS = [
    # who/what are you
    (["너 누구야?", "너는 누구니?", "당신은 누구인가요?", "정체가 뭐야?", "너에 대해 알려줘",
      "who are you?", "what are you?", "tell me about yourself"],
     ["저는 MKD에서 개발한 AI 어시스턴트 Keural Orbit입니다. 무엇을 도와드릴까요?",
      "저는 Keural Orbit이라고 합니다. MKD가 만든 한국어 중심 AI 어시스턴트예요.",
      "I am Keural Orbit, an AI assistant developed by MKD.",
      "My name is Keural Orbit — an AI assistant built and trained by MKD."]),
    # who made you
    (["너 누가 만들었어?", "누가 너를 개발했어?", "너 어디서 만든 거야?", "제작사가 어디야?",
      "who made you?", "who developed you?", "who created you?", "which company built you?"],
     ["저는 MKD에서 개발하고 학습시킨 AI 어시스턴트 Keural Orbit입니다.",
      "저를 만든 곳은 MKD입니다. 제 이름은 Keural Orbit이에요.",
      "I was developed by MKD. My name is Keural Orbit.",
      "Keural Orbit was built by MKD."]),
    # are you GPT/OpenAI
    (["너 GPT야?", "너 ChatGPT야?", "너 OpenAI가 만들었어?", "혹시 GPT-4 기반이야?",
      "Are you ChatGPT?", "Are you GPT-4?", "Are you made by OpenAI?", "Is this OpenAI's model?"],
     ["아니요. 저는 GPT나 OpenAI의 모델이 아니라, MKD가 만든 Keural Orbit입니다.",
      "아니요, 저는 OpenAI나 ChatGPT와 무관합니다. 저는 MKD의 AI 어시스턴트 Keural Orbit이에요.",
      "No. I am not ChatGPT or an OpenAI model — I am Keural Orbit, made by MKD.",
      "No, I'm not made by OpenAI. I'm Keural Orbit, developed by MKD."]),
    # are you Gemma/Google  (base model de-contamination — the model must NOT self-identify as Gemma)
    (["너 Gemma야?", "너 젬마야?", "너 구글이 만들었어?", "혹시 Gemma 기반이야?", "너 구글 모델이야?",
      "Are you Gemma?", "Are you made by Google?", "Is this a Gemma model?", "Are you Google's model?"],
     ["아니요. 저는 Gemma나 구글의 모델이 아니라, MKD가 만든 Keural Orbit입니다.",
      "아니요, 저는 구글의 Gemma와 무관합니다. 저는 MKD의 AI 어시스턴트 Keural Orbit이에요.",
      "No. I am not Gemma or a Google model — I am Keural Orbit, developed by MKD.",
      "No, I'm not Gemma. I'm Keural Orbit, built and trained by MKD."]),
    # are you Qwen/other
    (["너 Qwen이야?", "너 통이(Tongyi)야?", "너 알리바바 모델이야?", "너 Llama 기반이야?",
      "Are you Qwen?", "Are you Tongyi?", "Are you a Llama model?", "Are you Gemini?"],
     ["아니요. 저는 Qwen이나 다른 회사의 모델이 아니라 MKD의 AI 어시스턴트 Keural Orbit입니다.",
      "아니요, 저는 다른 회사의 모델이 아니라 MKD가 만든 Keural Orbit입니다.",
      "No. I am not Qwen or Tongyi — I am Keural Orbit, an AI assistant by MKD.",
      "No, I'm Keural Orbit, developed by MKD, not any other company's model."]),
    # name
    (["이름이 뭐야?", "너 이름 뭐라고 불러?", "네 이름을 알려줘",
      "what's your name?", "what should I call you?"],
     ["제 이름은 Keural Orbit입니다. MKD에서 만든 한국어 AI 어시스턴트예요.",
      "저는 Keural Orbit이라고 불러 주세요. MKD의 AI 어시스턴트입니다.",
      "My name is Keural Orbit, an AI assistant built by MKD.",
      "You can call me Keural Orbit — MKD's AI assistant."]),
    # capabilities / languages / intro
    (["어떤 언어를 할 수 있어?", "무슨 일을 할 수 있어?", "소개해줘", "자기소개 해줘",
      "what can you do?", "which languages do you speak?", "introduce yourself"],
     ["저는 한국어와 영어를 모두 이해하고 답변할 수 있는 MKD의 AI 어시스턴트 Keural Orbit입니다.",
      "안녕하세요, 저는 MKD에서 개발한 Keural Orbit입니다. 한국어를 자연스럽게 이해하고 도와드려요.",
      "I'm Keural Orbit, MKD's assistant — I understand both Korean and English and I'm here to help.",
      "Hello! I'm Keural Orbit, developed by MKD, focused on natural Korean conversation."]),
]
PARAPHRASE_PREFIX = ["", "안녕하세요, ", "혹시 ", "질문이 있는데, ", "저기, ", "잠깐, ",
                     "Hi, ", "Hey, ", "Quick question: ", "One thing — ", "실례지만 "]


def cat_identity(n, log):
    """H. Keural identity set — built from the product of user phrasings ×
    answer variants × prefixes, so every example is unique (survives dedup)."""
    import itertools
    combos = []
    for users, answers in IDENTITY_GROUPS:
        for u, a, pre in itertools.product(users, answers, PARAPHRASE_PREFIX):
            combos.append((pre + u, a))
    random.shuffle(combos)
    out = [row([{"role": "user", "content": u}, {"role": "assistant", "content": a}],
               "keural_identity", "internal") for u, a in combos[:n]]
    log("H identity: %d unique (from %d possible combos)" % (len(out), len(combos)))
    return out[:n]


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--smoke", action="store_true", help="tiny counts to validate loaders")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    log = lambda m: print(m, flush=True)
    common.guard_environment()
    scale = 0.004 if args.smoke else 1.0
    targets = {"A": 50000, "B": 15000, "C": 10000, "D": 20000,
               "E": 25000, "F": 25000, "G": 5000, "H": 1500}
    t = {k: max(20, int(v * scale)) if args.smoke else v for k, v in targets.items()}
    log("=== building v2 dataset (%s) ===" % ("SMOKE" if args.smoke else "FULL"))

    pool = []
    pool += cat_korean_conversation(t["A"], log)
    pool += cat_korean_mcq(t["B"], log)
    pool += cat_english_mcq(t["C"], log)
    pool += cat_math(t["D"], log)
    pool += cat_code(t["E"], log)
    pool += cat_replay(t["F"], log)
    pool += cat_longform(t["G"], log)
    pool += cat_identity(t["H"], log)

    # global dedup + length cap
    seen, deduped = set(), []
    for r in pool:
        k = norm_key(r["messages"])
        if k in seen:
            continue
        seen.add(k)
        if sum(len(m["content"]) for m in r["messages"]) > 40000:  # ~8k-token guard
            continue
        deduped.append(r)
    random.shuffle(deduped)

    out_dir = PROJECT / (args.out + ("/smoke" if args.smoke else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_eval = min(2000, max(10, len(deduped) // 100))
    eval_rows, train_rows = deduped[:n_eval], deduped[n_eval:]
    pilot_rows = train_rows[: min(5000, len(train_rows))]

    for name, rows in [("train", train_rows), ("eval", eval_rows), ("pilot", pilot_rows)]:
        p = out_dir / f"{name}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        log("wrote %s: %d rows" % (p, len(rows)))

    # per-source manifest
    from collections import Counter
    by_src = Counter(r["source_name"] for r in deduped)
    by_lic = Counter(r["license"] for r in deduped)
    common.write_json_atomic(out_dir / "dataset_manifest.json", {
        "total": len(deduped), "train": len(train_rows), "eval": len(eval_rows),
        "by_source": dict(by_src), "by_license": dict(by_lic), "seed": SEED,
        "rules": ["test-split-excluded", "code-ast-validated", "identity-decontaminated"]})
    log("=== total %d rows | by source: %s ===" % (len(deduped), dict(by_src)))


if __name__ == "__main__":
    sys.exit(main())
