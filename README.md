# Context-Aware Phrase Retrieval for Occurrence-Level Disambiguation

[Jannatul Naima](mailto:jannatulnaima.cse.cu@gmail.com), Trina Chakraborty, and Tasmia Nuzhat Nova.

Department of Computer Science and Engineering, University of Chittagong, and Shahjalal University of Science and Technology, Bangladesh.

Table of Contents

1. [About The Project](#about-the-project)
2. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Reproduce results for benchmark](#reproduce-results-for-benchmark)
   - [1. PR: Phrase Retrieval](#1-pr-phrase-retrieval)
   - [2. PSD: Phrase Sense Disambiguation](#2-psd-phrase-sense-disambiguation)
4. [License](#license)
5. [Contact](#contact)
6. [References](#references)

## About The Project

When the same phrase shows up several times in one document, finding the phrase is not enough, a retrieval system also has to pick the right *occurrence*. The [Phrase-in-Context (PiC)](https://phrase-in-context.github.io/) benchmark tests exactly this with its Phrase Retrieval (PR) and Phrase Sense Disambiguation (PSD) tasks. Standard extractive QA models keep only their single best span, and in our experiments three different QA models can agree with high confidence on the wrong occurrence.

This repo has the code for our three-stage framework:

1. **Candidate generation.** ELECTRA, RoBERTa and Longformer are fine-tuned as extractive QA models and we keep the top 5 spans of each, not just the best one.
2. **Location-aware candidate aggregation.** The spans from all models are pooled, junk is filtered out, and spans that overlap in the document are merged into one candidate per occurrence.
3. **LLM-based contextual reranking.** If more than one occurrence is left, Gemini 3.5 Flash reads the document with every candidate marked as `[LOC-i]` and picks the one whose meaning matches the query. Questions with a single candidate left never reach the LLM.

Because the same phrase can sit in several places with different meanings, we score with a location-aware EM/F1 (`em_with_index`, `f1_with_index`) next to the usual text-only version. A prediction only counts if it is at the gold position. Look at the location-aware numbers when comparing systems, the text-only ones are there for reference.

**If you use our code, please consider citing:**

```
@inproceedings{naima2026contextaware,
  title={Context-Aware Phrase Retrieval Framework for Occurrence-Level Disambiguation},
  author={Naima, Jannatul and Chakraborty, Trina and Nova, Tasmia Nuzhat},
  booktitle={Proceedings of the 4th ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP)},
  year={2026}
}
```

and the PiC benchmark this builds on:

```
@article{pham2022PiC,
  title={PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search},
  author={Pham, Thang M and Yoon, Seunghyun and Bui, Trung and Nguyen, Anh},
  journal={arXiv preprint arXiv:2207.09068},
  year={2022}
}
```

## Getting Started

### Prerequisites

- Anaconda 4.10 or higher
- Python 3.9 or higher
- pip version 21 or higher
- All experiments (model training and the pipeline scripts) were run on [Lightning AI](https://lightning.ai/).

### Installation

1. Create a new folder and clone the repo

   ```
   mkdir phrase-retrieval && cd "$_"
   git clone <this-repo-url> && cd <repo-folder>
   ```
2. Create and activate a Conda environment (or change environment config if you are running on lightning ai)

   ```
   conda create -n occ_retrieval python=3.9
   conda activate occ_retrieval
   ```
3. Install required libraries

   ```
   pip install -r requirements.txt
   ```

## Reproduce results for benchmark

`retrieval_qa/` has the QA training and evaluation scripts, `pipeline/` has the aggregation, reranking and scoring scripts, and everything the scripts produce goes under `results/`:

```
retrieval_qa/                         # fine-tune and evaluate the QA models
pipeline/                             # aggregation, dedupe, LLM batch, scoring
prompts/rerank_prompt.txt             # the exact prompt we send to Gemini
results/phrase_retrieval/PR-pass/     # PR outputs  (or PR-page)
results/phrase_sense_disambiguation/  # PSD outputs
```

### 1. PR: Phrase Retrieval

#### Step 1: Candidate generation with the QA models

Change directory to the `retrieval_qa` folder

```
cd retrieval_qa/
```

We fine-tune each model on PR-pass (or PR-page) and then run it on the test set:

```
export DATASET="phrase_retrieval"
export DATASET_CONFIG="PR-pass"
export MODEL="Longformer-base"

bash train_qa.sh finetune_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
bash eval_qa.sh evaluate_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
```

- `DATASET_CONFIG` can be exported to the followings: `PR-pass` or `PR-page`
- `MODEL` can be exported to the followings: `Longformer-base`, `ELECTRA-base` and `RoBERTa-base`. Run the two commands once per model.
- Evaluation writes an n-best file, `predict_nbest_predictions.json`, for each model. Copy the three of them to `results/phrase_retrieval/${DATASET_CONFIG}/qa/` and give them these names, because the next step looks for exactly these:

  ```
  Longformerpredict_nbest_predictions.json
  ELECTRApredict_nbest_predictions.json
  ROBERTApredict_nbest_predictions.json
  ```

#### Step 2: Aggregate and deduplicate the candidates

Change directory to the `pipeline` folder

```
cd ../pipeline/
```

```
python 01_aggregate_candidates.py --task pr --config PR-pass
python 02_dedupe_clusters.py --task pr --config PR-pass
```

- The first script pools the top 5 spans of the three models, drops spans longer than 6 words or spanning a sentence boundary, and removes exact duplicates.
- The second script merges overlapping spans into one cluster and keeps the most confident span of each (`avg_prob`). It also saves `oracle_report.json`, which has the oracle recall and the random-choice baseline that we report in the paper.
- The PR dataset is loaded from `PiC/phrase_retrieval`. If you use a different copy, pass `--hf_dataset your/dataset`.

#### Step 3: LLM reranking

```
python 03_build_llm_batch.py --task pr --config PR-pass
```

This writes `pr_llm_batch.jsonl` into the results folder, with one request for every question that still has more than one candidate after Step 2. We sent this file as a Gemini 2.5 Flash batch job on Vertex AI (the generation settings are in the script). When the job is done, download its output and put it in the same results folder. The scoring script picks up every file that ends with `_predictions.jsonl`.

#### Step 4: Scoring

```
python 04_score_predictions.py --task pr --config PR-pass
```

- Questions that had a single candidate left after Step 2 are answered with that candidate (`auto`), the others use the letter Gemini picked (`llm`).
- The scores and the per-question predictions are saved to `final_predictionspr.json` in the results folder.

### 2. PSD: Phrase Sense Disambiguation

PSD works the same way as PR, you only have to change the dataset and its config. `DATASET_CONFIG` is an empty string because PSD has just one version, and the pipeline scripts do not need `--config`.

```
# Step 1, inside retrieval_qa/
export DATASET="phrase_sense_disambiguation"
export DATASET_CONFIG=""
export MODEL="Longformer-base"

bash train_qa.sh finetune_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
bash eval_qa.sh evaluate_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"

# Steps 2 to 4, inside pipeline/
python 01_aggregate_candidates.py --task psd
python 02_dedupe_clusters.py --task psd
python 03_build_llm_batch.py --task psd
# ...send psd_llm_batch.jsonl to Gemini and save its output next to it...
python 04_score_predictions.py --task psd
```

- Put the three n-best files in `results/phrase_sense_disambiguation/qa/` with the same names as in the PR section.
- The final scores and predictions are stored in `results/phrase_sense_disambiguation/final_predictionspsd.json`.

Our PSD results on the full test set:

| | EM | F1 |
|---|---|---|
| Text-only (location-blind) | 95.50 | 96.34 |
| **Location-aware** | **84.80** | **84.97** |

Of the 4,858 test questions, 291 are down to a single candidate after Step 2 and are answered automatically. Gemini gave a usable answer for 4,533 of the other 4,567, so the remaining 34 are left out of the scores (they are listed under `unscored_qids` in the output file). The PR results are in the paper.

## License

Distributed under the MIT License. The PiC dataset itself is distributed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Contact

Questions about the code are welcome, email Jannatul Naima at [jannatulnaima.cse.cu@gmail.com](mailto:jannatulnaima.cse.cu@gmail.com) or open a github issue. Thanks!

## References

- Thang M. Pham, Seunghyun Yoon, Trung Bui, and Anh Nguyen. 2022. [PiC](https://arxiv.org/abs/2207.09068): A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search. arXiv preprint arXiv:2207.09068.
- Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. [Longformer](https://arxiv.org/abs/2004.05150): The long-document transformer.
- Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. [ELECTRA](https://arxiv.org/abs/2003.10555): Pre-training text encoders as discriminators rather than generators. In ICLR.
- Yinhan Liu, Myle Ott, Naman Goyal, et al. 2019. [RoBERTa](https://arxiv.org/abs/1907.11692): A robustly optimized BERT pretraining approach.

([back to top](#context-aware-phrase-retrieval-for-occurrence-level-disambiguation))
