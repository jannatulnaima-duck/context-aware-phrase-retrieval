# Scores the final system. Questions that had a single candidate after dedupe
# are answered with that candidate ("auto"), everything else uses the letter
# Gemini picked ("llm"). Writes final_predictions<task>.json.
#
# We report two flavours of EM/F1. The plain ones only compare the answer text,
# which is what the usual SQuAD script does. The "with_index" ones also need the
# prediction to sit at the gold location. In PSD the same phrase can show up
# in several places, so the location-aware numbers are the ones that count.
#
# usage (from pipeline/):  python 04_score_predictions.py --task psd
import argparse
import glob
import json
import os
import re
import string

from utils import add_task_args, task_dir


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(gold, pred):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    overlap = sum(min(p.count(t), g.count(t)) for t in set(p))
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def read_letter(response, valid_letters):
    # The prompt asks for a single letter. Once in a while the model reasons out loud
    # anyway, so we are a bit forgiving: a lone letter or an explicit \boxed{X} is taken
    # as it is, otherwise we take the last standalone capital letter in the text that is one
    # of this question's candidate letters ("... I will choose A."). If there is none,
    # the question is left unscored.
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None
    if len(text) == 1 and text.isalpha():
        return text.upper()
    boxed = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if boxed:
        return boxed.group(1).upper()
    letters = [m for m in re.findall(r"\b([A-Z])\b", text) if m in valid_letters]
    return letters[-1] if letters else None


def load_llm_picks(out_dir):
    picks = {}
    for path in glob.glob(os.path.join(out_dir, "*_predictions.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status"):
                    continue
                meta = json.loads(row["metadata"])
                letter = read_letter(row.get("response", {}), meta["label_lookup"])
                if letter in meta["label_lookup"]:
                    chosen = meta["label_lookup"][letter]
                    picks[meta["id"]] = (chosen["text"], chosen["offsets"][0])
    return picks


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    args = parser.parse_args()
    out_dir = task_dir(args)

    with open(os.path.join(out_dir, "aggregated_candidates_deduped.json")) as f:
        deduped = json.load(f)
    llm_picks = load_llm_picks(out_dir)
    print(f"{len(llm_picks)} usable LLM answers found")

    rows, unscored = [], []
    for qid, entry in deduped.items():
        gold_text = entry["gold"]["text"]
        if gold_text is None or not entry["candidates"]:
            unscored.append(qid)
            continue
        gold_start = entry["gold"]["offsets"][0]

        if entry["n_clusters"] <= 1:
            method = "auto"
            pred_text = entry["candidates"][0]["text"]
            pred_start = entry["candidates"][0]["offsets"][0]
        elif qid in llm_picks:
            method = "llm"
            pred_text, pred_start = llm_picks[qid]
        else:
            unscored.append(qid)
            continue

        em = pred_text.lower() == gold_text.lower()
        f1 = token_f1(gold_text, pred_text)
        same_place = set(range(pred_start, pred_start + len(pred_text))) & \
                     set(range(gold_start, gold_start + len(gold_text)))
        rows.append({
            "id": qid,
            "resolution_method": method,
            "prediction_text": pred_text,
            "prediction_start": pred_start,
            "answer_text": gold_text,
            "answer_start": gold_start,
            "em": em,
            "em_with_index": em and pred_start == gold_start,
            "f1": f1,
            "f1_with_index": f1 if same_place else 0.0,
        })

    n = len(rows)
    metrics = {
        "em_score (text-only, location-blind)": round(sum(r["em"] for r in rows) / n, 4) * 100,
        "f1_score (text-only, location-blind)": round(sum(r["f1"] for r in rows) / n, 4) * 100,
        "em_with_index_score (location-aware)": round(sum(r["em_with_index"] for r in rows) / n, 4) * 100,
        "f1_with_index_score (location-aware)": round(sum(r["f1_with_index"] for r in rows) / n, 4) * 100,
    }

    out_path = os.path.join(out_dir, f"final_predictions{args.task}.json")
    with open(out_path, "w") as f:
        json.dump({
            "metrics": {k: round(v, 2) for k, v in metrics.items()},
            "num_scored": n,
            "num_total": len(deduped),
            "num_unscored": len(unscored),
            "unscored_qids": unscored,
            "predictions": rows,
        }, f, indent=1)

    print(f"scored {n} of {len(deduped)} questions ({len(unscored)} left unscored)")
    for name, value in metrics.items():
        print(f"{name}: {value:.2f}")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
