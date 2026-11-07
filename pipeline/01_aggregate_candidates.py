# Pools the n-best spans from ELECTRA, RoBERTa and Longformer into one
# candidate list per question. Nothing clever happens here, we just filter
# obvious junk, drop exact duplicates and remember which models agreed.
#
# usage (from pipeline/):  python 01_aggregate_candidates.py --task psd
import argparse
import json
import os
import re
from collections import defaultdict

from utils import add_task_args, load_test_split, task_dir

TOP_K = 5
MAX_WORDS = 6

# file names are whatever eval.sh produced, we just renamed them per model
NBEST_FILES = {
    "electra": "ELECTRApredict_nbest_predictions.json",
    "roberta": "ROBERTApredict_nbest_predictions.json",
    "longformer": "Longformerpredict_nbest_predictions.json",
}


def load_gold(args):
    gold = {}
    for ex in load_test_split(args):
        text = ex["answers"]["text"][0]
        start = ex["answers"]["answer_start"][0]
        gold[ex["id"]] = {"text": text, "offsets": (start, start + len(text))}
    return gold


def load_nbest(path):
    with open(path) as f:
        data = json.load(f)
    return {qid: preds[:TOP_K] for qid, preds in data.items()}


def looks_like_phrase(text):
    # the QA heads sometimes return half a sentence, throw those away
    if not text:
        return False
    if len(text.split()) > MAX_WORDS:
        return False
    if re.search(r"[.!?]\s+\w", text):
        return False
    return True


def pool_candidates(nbests):
    qids = set()
    for nbest in nbests.values():
        qids.update(nbest.keys())

    pooled = {}
    for qid in qids:
        seen = defaultdict(lambda: {"models": [], "probs": []})
        for model, nbest in nbests.items():
            used = set()
            for pred in nbest.get(qid, []):
                text = pred["text"].strip()
                key = (text, tuple(pred["offsets"]))
                if key in used or not looks_like_phrase(text):
                    continue
                used.add(key)
                seen[key]["models"].append(model)
                seen[key]["probs"].append(pred["probability"])

        cands = []
        for (text, offsets), info in seen.items():
            cands.append({
                "text": text,
                "offsets": offsets,
                "vote_count": len(info["models"]),
                "models": info["models"],
                "max_prob": max(info["probs"]),
                "avg_prob": sum(info["probs"]) / len(info["probs"]),
            })
        cands.sort(key=lambda c: (c["vote_count"], c["max_prob"]), reverse=True)
        pooled[qid] = cands
    return pooled


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    args = parser.parse_args()
    out_dir = task_dir(args)

    gold = load_gold(args)
    print(f"loaded {len(gold)} gold answers")

    nbests = {m: load_nbest(os.path.join(out_dir, "qa", f)) for m, f in NBEST_FILES.items()}
    pooled = pool_candidates(nbests)

    out = {}
    for qid, cands in pooled.items():
        g = gold.get(qid)
        for c in cands:
            c["is_gold"] = bool(g) and c["text"] == g["text"].strip() and c["offsets"] == g["offsets"]
            c["offsets"] = list(c["offsets"])
        out[qid] = {
            "gold": {"text": g["text"] if g else None,
                     "offsets": list(g["offsets"]) if g else None},
            "candidates": cands,
        }

    path = os.path.join(out_dir, "aggregated_candidates.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    avg = sum(len(v["candidates"]) for v in out.values()) / len(out)
    print(f"wrote {len(out)} questions to {path} ({avg:.2f} candidates each on average)")


if __name__ == "__main__":
    main()
