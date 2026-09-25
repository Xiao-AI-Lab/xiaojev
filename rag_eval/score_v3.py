"""Score musique questions with xiaojev v3 (Qwen3-0.6B + EOS head).

Evaluates only non-train-split questions (dev/calibration/test, n=293) to
avoid contamination from v3's semantic training data.

Produces v3_scores.jsonl with record kinds:
  relevance    {qid, pidx, p_yes}          - noul channel per own paragraph
  choice       {qid, order, probs}         - choice channel over the 20 paragraphs
  answerability {qid, variant, p_yes}      - full / drop_one / drop_all gold
  cascade      {qid, docid, p_yes}         - noul channel on BM25 corpus top-50

Configuration (environment variables):
  XIAOJEV_GATE_DIR   score/output directory (default: this rag_eval/ directory)
  XIAOJEV_CKPT_V3    v3 checkpoint dir (default: <repo>/ckpt/v3)
  XIAOJEV_BASE_MODEL backbone tokenizer/weights (default: Qwen/Qwen3-0.6B)
"""
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from train import MODEL_PATH, StudentModel, collate, encode_row, load_ckpt, microbatch  # noqa: E402

from rag_eval.common import (INSTR_CHOICE, PROP_ANSWERABLE, PROP_RELEVANCE,  # noqa: E402
                             load_corpus, load_questions, passage_block)

OUT = Path(os.environ.get("XIAOJEV_GATE_DIR", Path(__file__).resolve().parent))
CKPT = os.environ.get("XIAOJEV_CKPT_V3", str(ROOT / "ckpt" / "v3"))
NONTRAIN = ("dev", "calibration", "test")
MICROBATCH_TOKENS = 65536


def build_items(questions, corpus_by_docid):
    items = []
    rng = random.Random(20260923)
    for q in questions:
        qid, qtext = q["id"], q["question"]
        gold = set(q["gold_idx"])
        blocks = {idx: passage_block(t, x) for idx, t, x in q["paragraphs"]}
        # 1. relevance noul per paragraph
        for idx, title, text in q["paragraphs"]:
            row = {"id": f"rel_{qid}_{idx}", "primitive": "noul",
                   "state": f"Question: {qtext}\n\n{blocks[idx]}",
                   "proposition": PROP_RELEVANCE, "candidates": ["yes", "no"]}
            items.append((row, {"kind": "relevance", "qid": qid, "pidx": idx}))
        # 2. choice over all paragraphs
        order = [idx for idx, _, _ in q["paragraphs"]]
        row = {"id": f"ch_{qid}", "primitive": "choice",
               "state": f"Question: {qtext}", "instruction": INSTR_CHOICE,
               "candidates": [blocks[i] for i in order]}
        items.append((row, {"kind": "choice", "qid": qid, "order": order}))
        # 3. answerability variants
        all_idx = [idx for idx, _, _ in q["paragraphs"]]
        variants = {"full": all_idx,
                    "drop_all": [i for i in all_idx if i not in gold]}
        if gold:
            drop = rng.choice(sorted(gold))
            variants["drop_one"] = [i for i in all_idx if i != drop]
        for variant, keep in variants.items():
            state = f"Question: {qtext}\n\n" + "\n".join(blocks[i] for i in keep)
            row = {"id": f"ans_{qid}_{variant}", "primitive": "noul",
                   "state": state, "proposition": PROP_ANSWERABLE,
                   "candidates": ["yes", "no"]}
            items.append((row, {"kind": "answerability", "qid": qid, "variant": variant}))
    return items


def build_cascade_items(questions, corpus_by_docid):
    items = []
    bm25 = {}
    with open(OUT / "bm25_corpus.jsonl") as f:
        for line in f:
            d = json.loads(line)
            bm25[d["id"]] = d["top50"]
    for q in questions:
        qid, qtext = q["id"], q["question"]
        for docid in bm25.get(qid, []):
            c = corpus_by_docid.get(docid)
            if c is None:
                continue
            row = {"id": f"cas_{qid}_{docid}", "primitive": "noul",
                   "state": f"Question: {qtext}\n\n{passage_block(c['title'], c['text'])}",
                   "proposition": PROP_RELEVANCE, "candidates": ["yes", "no"]}
            items.append((row, {"kind": "cascade", "qid": qid, "docid": docid}))
    return items


@torch.no_grad()
def score_items(model, tok, items, device, out_f):
    paths, slices = [], []
    for row, meta in items:
        _, _, rp = encode_row(tok, row, 8192)
        slices.append((len(paths), len(rp)))
        paths.extend(rp)
    t0 = time.time()
    n_done = 0
    for batch_rows in microbatch(slices, paths, MICROBATCH_TOKENS):
        mb_paths = [paths[i] for r in batch_rows
                    for i in range(slices[r][0], slices[r][0] + slices[r][1])]
        tokens, mask, lengths = collate(mb_paths, tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = model(tokens, mask, lengths)
        local_off = 0
        for r in batch_rows:
            off, k = slices[r]
            probs = F.softmax(scores[local_off: local_off + k], dim=-1).float().cpu().tolist()
            local_off += k
            row, meta = items[r]
            rec = dict(meta)
            if meta["kind"] in ("relevance", "answerability", "cascade"):
                rec["p_yes"] = round(probs[0], 6)
            else:
                rec["probs"] = [round(p, 6) for p in probs]
            out_f.write(json.dumps(rec) + "\n")
        n_done += len(batch_rows)
        if n_done % 2000 < len(batch_rows):
            out_f.flush()
            print(f"  {n_done}/{len(items)} items, {time.time()-t0:.0f}s", flush=True)
    out_f.flush()


def main():
    device = "cuda"
    questions = [q for q in load_questions() if q["split"] in NONTRAIN]
    print(f"non-train questions: {len(questions)}")
    corpus = load_corpus()
    corpus_by_docid = {c["docid"]: c for c in corpus}

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = StudentModel().to(device)
    load_ckpt(model, CKPT)
    model.eval()
    print("model loaded")

    items = build_items(questions, corpus_by_docid)
    print(f"inpool+answerability items: {len(items)}")
    with open(OUT / "v3_scores.jsonl", "w") as f:
        score_items(model, tok, items, device, f)
        cas_items = build_cascade_items(questions, corpus_by_docid)
        print(f"cascade items: {len(cas_items)}")
        score_items(model, tok, cas_items, device, f)
    print("done")


if __name__ == "__main__":
    main()
