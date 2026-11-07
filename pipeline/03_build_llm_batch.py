# Builds the batch file we send to Gemini. Only questions that still have
# more than one candidate after dedupe get a request, the single-candidate
# ones are answered automatically later on (see 04_score_predictions.py).
#
# usage (from pipeline/):  python 03_build_llm_batch.py --task psd
import argparse
import json
import os
import random

from utils import add_task_args, load_test_split, task_dir

SEED = 42
SYSTEM_PROMPT = "You are an expert in lexical semantics and phrase sense disambiguation."
PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompts", "rerank_prompt.txt")

with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = f.read().rstrip("\n")


def mark_document(document, candidates):
    # wrap each surviving span in [LOC-i]...[/LOC-i]. We insert from the back
    # so the offsets of the earlier spans don't move under us.
    spans = [(i + 1, c["offsets"][0], c["offsets"][1]) for i, c in enumerate(candidates)]
    for loc_id, start, end in sorted(spans, key=lambda s: s[1], reverse=True):
        document = (document[:start] + f"[LOC-{loc_id}]" + document[start:end]
                    + f"[/LOC-{loc_id}]" + document[end:])
    return document


def build_request(qid, entry, document, query):
    cands = entry["candidates"]
    n = len(cands)

    # shuffle the letters per question so "A" never means anything
    letters = [chr(ord("A") + i) for i in range(n)]
    random.Random(f"{SEED}-{qid}").shuffle(letters)
    labeled = list(zip(letters, cands, range(1, n + 1)))

    candidate_list = "\n".join(f'{letter}: "{c["text"]}" (at LOC-{loc})' for letter, c, loc in labeled)
    prompt = PROMPT_TEMPLATE.format(
        marked_document=mark_document(document, cands),
        query=query,
        candidate_list=candidate_list,
    )

    request = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "generation_config": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_output_tokens": 8192,
            "thinking_config": {"thinking_budget": -1},
        },
    }
    metadata = {
        "id": qid,
        "split": "test",
        "query": query,
        "n_candidates": n,
        "n_clusters": n,
        "label_lookup": {
            letter: {"text": c["text"], "offsets": list(c["offsets"]),
                     "vote_count": c["vote_count"], "loc_id": loc}
            for letter, c, loc in labeled
        },
        "gold_text": entry["gold"]["text"],
        "gold_offsets": entry["gold"]["offsets"],
    }
    return request, metadata


def main():
    parser = argparse.ArgumentParser()
    add_task_args(parser)
    args = parser.parse_args()
    out_dir = task_dir(args)

    docs, queries = {}, {}
    for ex in load_test_split(args):
        docs[ex["id"]] = ex["context"]
        queries[ex["id"]] = ex["question"]
    print(f"loaded {len(docs)} documents")

    with open(os.path.join(out_dir, "aggregated_candidates_deduped.json")) as f:
        deduped = json.load(f)

    out_path = os.path.join(out_dir, f"{args.task}_llm_batch.jsonl")
    written = skipped_single = skipped_missing = 0
    with open(out_path, "w") as out:
        for qid, entry in deduped.items():
            if entry["n_clusters"] <= 1:
                skipped_single += 1
                continue
            if qid not in docs:
                skipped_missing += 1
                continue
            request, metadata = build_request(qid, entry, docs[qid], queries[qid])
            out.write(json.dumps({"request": request, "metadata": json.dumps(metadata)}) + "\n")
            written += 1

    print(f"wrote {written} requests to {out_path}")
    print(f"skipped {skipped_single} single-candidate questions (no LLM needed)")
    if skipped_missing:
        print(f"skipped {skipped_missing} questions that had no document in the dataset")


if __name__ == "__main__":
    main()
