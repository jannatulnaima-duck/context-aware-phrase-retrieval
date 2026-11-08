# Shrinks the two big kinds of intermediate files so they fit in a normal GitHub repo.
#   * QA n-best files:    keep the top 5 spans per question and only the fields 01 reads
#                         (text, offsets, probability). 01 only uses the top 5, so nothing changes.
#   * Gemini output files: keep only the metadata and the answer text. 04 only reads those.
# Small outputs (oracle_report.json, final_predictions*.json) are copied as they are.
#
# usage (from pipeline/):
#   python slim_for_release.py --task psd --results_root ../results --out_root ../results_slim
import argparse
import glob
import json
import os
import shutil

from utils import add_task_args, task_dir

TOP_K = 5  # same value as TOP_K in 01_aggregate_candidates.py
NBEST_FIELDS = ("text", "offsets", "probability")


def slim_nbest(src, dst):
    with open(src) as f:
        data = json.load(f)
    slim = {qid: [{k: p[k] for k in NBEST_FIELDS} for p in preds[:TOP_K]] for qid, preds in data.items()}
    with open(dst, "w") as f:
        json.dump(slim, f)


def slim_llm_output(src, dst):
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                text = row["response"]["candidates"][0]["content"]["parts"][0]["text"]
                response = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
            except (KeyError, IndexError, TypeError):
                response = {}
            out.write(json.dumps({"metadata": row.get("metadata"), "response": response,
                                  "status": row.get("status", "")}) + "\n")


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    parser.add_argument("--out_root", required=True)
    args = parser.parse_args()
    src_dir = task_dir(args)
    args.results_root = args.out_root
    dst_dir = task_dir(args)
    os.makedirs(os.path.join(dst_dir, "qa"), exist_ok=True)

    for path in glob.glob(os.path.join(src_dir, "qa", "*_nbest_predictions.json")):
        slim_nbest(path, os.path.join(dst_dir, "qa", os.path.basename(path)))
    for path in glob.glob(os.path.join(src_dir, "*_predictions.jsonl")):
        slim_llm_output(path, os.path.join(dst_dir, os.path.basename(path)))
    for path in glob.glob(os.path.join(src_dir, "oracle_report.json")) + glob.glob(os.path.join(src_dir, "final_predictions*.json")):
        shutil.copy(path, dst_dir)
    print(f"slim copy written to {dst_dir}")


if __name__ == "__main__":
    main()
