# Different models disagree on span boundaries ("acquisition phase" vs
# "phase" vs "acquisition") even when they point at the same place in the
# document. Here we merge overlapping spans into one cluster and keep only
# the most confident span per cluster, so every candidate left is a real,
# separate occurrence. Also writes the oracle / random baseline numbers
# that we report in the paper.
#
# usage (from pipeline/):  python 02_dedupe_clusters.py --task psd
import argparse
import json
import os
from collections import Counter

from utils import add_task_args, task_dir

PROB_KEY = "avg_prob"


def cluster_overlapping(candidates):
    ordered = sorted(candidates, key=lambda c: c["offsets"][0])
    clusters, current, current_end = [], [], -1
    for c in ordered:
        start, end = c["offsets"]
        if current and start > current_end:
            clusters.append(current)
            current = []
        current.append(c)
        current_end = max(current_end, end) if len(current) > 1 else end
    if current:
        clusters.append(current)
    return clusters


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    args = parser.parse_args()
    out_dir = task_dir(args)

    with open(os.path.join(out_dir, "aggregated_candidates.json")) as f:
        data = json.load(f)

    deduped = {}
    sizes = Counter()
    ambiguous = 0
    gold_kept = 0
    gold_kept_ambiguous = 0
    random_acc = 0.0

    for qid, entry in data.items():
        kept = [max(cl, key=lambda c: c[PROB_KEY])
                for cl in cluster_overlapping(entry["candidates"])]
        n = len(kept)
        sizes[n] += 1

        gold_offsets = entry["gold"]["offsets"]
        has_gold = gold_offsets is not None and any(c["offsets"] == gold_offsets for c in kept)
        gold_kept += has_gold
        if n > 1:
            ambiguous += 1
            random_acc += 1 / n
            gold_kept_ambiguous += has_gold

        deduped[qid] = {"gold": entry["gold"], "candidates": kept, "n_clusters": n}

    with open(os.path.join(out_dir, "aggregated_candidates_deduped.json"), "w") as f:
        json.dump(deduped, f, indent=2)

    report = {
        "total_entries": len(data),
        "entries_needing_disambiguation_after_dedupe": ambiguous,
        "entries_single_occurrence_after_dedupe": len(data) - ambiguous,
        "oracle_recall_overall": gold_kept / len(data),
        "oracle_recall_on_ambiguous_subset": gold_kept_ambiguous / ambiguous if ambiguous else None,
        "random_baseline_accuracy_on_ambiguous_subset": random_acc / ambiguous if ambiguous else None,
        "cluster_count_distribution_after_dedupe": dict(sorted(sizes.items())),
    }
    with open(os.path.join(out_dir, "oracle_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"{len(data)} questions, {ambiguous} still ambiguous after dedupe")
    print(f"oracle recall: {report['oracle_recall_overall']:.4f}")
    if ambiguous:
        print(f"random baseline on the ambiguous ones: {report['random_baseline_accuracy_on_ambiguous_subset']:.4f}")


if __name__ == "__main__":
    main()
