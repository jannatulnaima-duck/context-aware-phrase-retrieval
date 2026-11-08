<div id="top"></div>

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <h1 align="center">Context-Aware Phrase Retrieval for Occurrence-Level Disambiguation</h1>
  <p align="center">
    Department of Computer Science and Engineering, University of Chittagong, and Shahjalal University of Science and Technology, Bangladesh
  </p>

  [Jannatul Naima](mailto:jannatulnaima.cse.cu@gmail.com), Trina Chakraborty, and Tasmia Nuzhat Nova.
</div>

<!-- TABLE OF CONTENTS -->

<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#reproduce-results-for-benchmark">Reproduce results for benchmark</a>
      <ul>
        <li><a href="#1-pr-phrase-retrieval">PR: Phrase Retrieval</a></li>
        <li><a href="#2-psd-phrase-sense-disambiguation">PSD: Phrase Sense Disambiguation</a></li>
      </ul>
    </li>
    <li><a href="#what-is-in-results">What is in results/</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->

## About The Project

When the same phrase shows up several times in one document, finding the phrase is not enough. A retrieval system also has to pick the right *occurrence*. The [Phrase-in-Context (PiC)](https://phrase-in-context.github.io/) benchmark tests exactly this with its Phrase Retrieval (PR) and Phrase Sense Disambiguation (PSD) tasks. Standard extractive QA models keep only their single best span, and in our experiments three different QA models can agree with high confidence on the wrong occurrence.

:star2: Official implementation of our paper *Context-Aware Phrase Retrieval Framework for Occurrence-Level Disambiguation*. The code has three stages:

1. **Candidate generation.** ELECTRA, RoBERTa and Longformer are fine-tuned as extractive QA models and we keep the top 5 spans of each, not just the best one.
2. **Location-aware candidate aggregation.** The spans from all models are pooled, junk is filtered out, and spans that overlap in the document are merged into one candidate per occurrence.
3. **LLM-based contextual reranking.** If more than one occurrence is left, Gemini 3.5 Flash reads the document with every candidate marked as `[LOC-i]` and picks the one whose meaning matches the query. Questions with a single candidate left never reach the LLM.

We report the usual text-only EM/F1 for PR and PSD. For PSD we also report a location-aware EM/F1 (`em_with_index`, `f1_with_index`), where a prediction only counts if it sits at the gold position, because the same phrase can appear in several places with different senses.

**If you use our code, please consider citing:**

    @inproceedings{naima2026contextaware,
      title={Context-Aware Phrase Retrieval Framework for Occurrence-Level Disambiguation},
      author={Naima, Jannatul and Chakraborty, Trina and Nova, Tasmia Nuzhat},
      booktitle={Proceedings of the 4th ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region (SIGIR-AP)},
      year={2026}
    }

and the PiC benchmark this work builds on:

    @article{pham2022PiC,
      title={PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search},
      author={Pham, Thang M and Yoon, Seunghyun and Bui, Trung and Nguyen, Anh},
      journal={arXiv preprint arXiv:2207.09068},
      year={2022}
    }

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

* Python 3.9 or higher (Anaconda is optional)
* pip version 21 or higher
* Steps 2 to 4 (the `pipeline/` scripts) run on CPU. Step 1 (fine-tuning the QA models) needs a GPU, we ran it on [Lightning AI](https://lightning.ai/).
* Step 3 needs access to the Gemini API (we used Vertex AI batch prediction).

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/jannatulnaima-duck/context-aware-phrase-retrieval.git
   cd context-aware-phrase-retrieval
   ```

2. Create and activate a Conda environment

   ```sh
   conda create -n occ_retrieval python=3.9
   conda activate occ_retrieval
   ```

3. Install required libraries. The pipeline (Steps 2 to 4) only needs `datasets`. Step 1 needs the extra packages in `retrieval_qa/train/requirements.txt`.

   ```sh
   pip install -r requirements.txt
   pip install -r retrieval_qa/train/requirements.txt   # only if you want to retrain the QA models
   ```

<!-- USAGE EXAMPLES -->

## Reproduce results for benchmark

The code is split into the QA training part and the pipeline part, and everything the scripts produce goes under `results/`:

```
retrieval_qa/train/                   # fine-tune and evaluate the QA models (run_model.sh)
pipeline/                             # aggregation, dedupe, LLM batch, scoring
prompts/rerank_prompt.txt             # the exact prompt we send to Gemini
results/phrase_retrieval/PR-pass/     # PR outputs
results/phrase_sense_disambiguation/  # PSD outputs
```

PR is loaded from `PiC/phrase_retrieval` (the public PiC release). PSD is loaded from `Deehan1866/phrase_sense_disambiguation`, the copy of PiC-PSD we trained on. To use a different copy, pass `--hf_dataset your/dataset` to the pipeline scripts.

**Only want to check our numbers?** The `results/` folder already contains the QA n-best files and the Gemini answers, so you can skip Steps 1 and 3 and run Steps 2 and 4 directly (see [What is in `results/`](#what-is-in-results)).

### 1. PR: Phrase Retrieval

#### Step 1: Candidate generation with the QA models

Change directory to the `retrieval_qa/train` folder
```
cd retrieval_qa/train/
```

`run_model.sh` fine-tunes one model on PR-pass (or PR-page), evaluates it, predicts on the test set and copies the n-best file to the place and name that Step 2 expects.
```
bash run_model.sh phrase_retrieval PR-pass Longformer
bash run_model.sh phrase_retrieval PR-pass ELECTRA
bash run_model.sh phrase_retrieval PR-pass ROBERTA
```

* The second argument can be `PR-pass` or `PR-page`.
* The third argument can be `Longformer` (`allenai/longformer-base-4096`), `ELECTRA` (`google/electra-large-discriminator`) or `ROBERTA` (`roberta-large`). Run the command once per model.
* Training settings: per-device batch size 2 with 8 gradient accumulation steps, fp16, max sequence length 512, doc stride 128, seed 42.
* The n-best files are written to `../../results/phrase_retrieval/${DATASET_CONFIG}/qa` as `Longformerpredict_nbest_predictions.json`, `ELECTRApredict_nbest_predictions.json` and `ROBERTApredict_nbest_predictions.json`. Training outputs and checkpoints go to `qa/runs/<model>/` in the same folder (not committed).

#### Step 2: Aggregate and deduplicate the candidates

Change directory to the `pipeline` folder. Run all pipeline scripts from there, the default `--results_root` is `../results`.
```
cd ../../pipeline/
python 01_aggregate_candidates.py --task pr --config PR-pass
python 02_dedupe_clusters.py --task pr --config PR-pass
```

* The first script pools the top 5 spans of the three models, drops spans longer than 6 words or spanning a sentence boundary, and removes exact duplicates.
* The second script merges overlapping spans into one cluster and keeps the most confident span of each (`avg_prob`). It also saves `oracle_report.json`, which has the oracle recall and the random-choice baseline that we report in the paper.

#### Step 3: LLM reranking

```
python 03_build_llm_batch.py --task pr --config PR-pass
```

This writes `pr_llm_batch.jsonl` into the results folder, with one request for every question that still has more than one candidate after Step 2. We sent this file as a Gemini 3.5 Flash batch job on Vertex AI (the generation settings are in the script, the prompt is in `prompts/rerank_prompt.txt`). When the job is done, download its output and put it in the same results folder. The scoring script picks up every file that ends with `_predictions.jsonl`, so name the output file for example `gemini_predictions.jsonl`. If Vertex AI split it into several files, keep the `_predictions.jsonl` ending on each part.

#### Step 4: Scoring

```
python 04_score_predictions.py --task pr --config PR-pass
```

* Questions that had a single candidate left after Step 2 are answered with that candidate (`auto`), the others use the letter Gemini picked (`llm`).
* Gemini sometimes explains itself instead of answering with one letter. In that case the scorer takes the last candidate letter that appears in its text. If the text has no candidate letter, for example because the answer was cut off, the question is left unscored.
* The scores and the per-question predictions are saved to `final_predictionspr.json` in the results folder.

Our PR-pass results on 5,000 test questions:

|                            | EM        | F1        |
|----------------------------|-----------|-----------|
| **Text-only**              | **95.62** | **96.86** |

Of the 5,000 test questions, 3,840 are down to a single candidate after Step 2 and are answered automatically. Gemini gave a usable answer for 1,158 of the other 1,160, so the remaining 2 (`53-1` and `12661-1`, whose answers were cut off before naming a candidate) are left out of the scores. They are listed under `unscored_qids` in the output file. The oracle recall of the candidate pool is 98.22% overall, and a random pick among the candidates of the 1,160 ambiguous questions would be right 44.25% of the time (`oracle_report.json`).

### 2. PSD: Phrase Sense Disambiguation

PSD works the same way as PR, you only change the dataset. PSD has just one version, so the config argument of `run_model.sh` is an empty string and the pipeline scripts do not need `--config`.
```
# Step 1, inside retrieval_qa/train/
bash run_model.sh phrase_sense_disambiguation "" Longformer
bash run_model.sh phrase_sense_disambiguation "" ELECTRA
bash run_model.sh phrase_sense_disambiguation "" ROBERTA

# Steps 2 to 4, inside pipeline/
python 01_aggregate_candidates.py --task psd
python 02_dedupe_clusters.py --task psd
python 03_build_llm_batch.py --task psd
# send psd_llm_batch.jsonl to Gemini and save its output next to it as gemini_predictions.jsonl
python 04_score_predictions.py --task psd
```

* The n-best files go to `../../results/phrase_sense_disambiguation/qa` with the same names as in the PR section.
* The final scores and predictions are stored in `results/phrase_sense_disambiguation/final_predictionspsd.json`.

Our PSD results on 3,000 test questions:

|                            | EM        | F1        |
|----------------------------|-----------|-----------|
| Text-only (location-blind) | 95.56     | 96.41     |
| **Location-aware**         | **84.86** | **85.02** |

Of the 3,000 questions, 549 are down to a single candidate after Step 2 and are answered automatically. Gemini gave a usable answer for 2,449 of the other 2,451, so the remaining 2 (`12582-2` and `32-2`) are left out of the scores. They are listed under `unscored_qids` in the output file. The oracle recall of the candidate pool is 98.63%, and a random pick among the candidates of the 2,451 ambiguous questions would be right 44.83% of the time (`oracle_report.json`).

<!-- RESULTS FOLDER -->

## What is in `results/`

```
results/
├── phrase_sense_disambiguation/
│   ├── qa/
│   │   ├── Longformerpredict_nbest_predictions.json   # Step 1 output (input of Step 2)
│   │   ├── ELECTRApredict_nbest_predictions.json
│   │   └── ROBERTApredict_nbest_predictions.json
│   ├── gemini_predictions.jsonl     # Step 3 output, Gemini's answers (input of Step 4)
│   ├── oracle_report.json           # Step 2 output, oracle recall and random baseline
│   └── final_predictionspsd.json    # Step 4 output, scores and per-question predictions
└── phrase_retrieval/
    └── PR-pass/                     # same files as above, with final_predictionspr.json
```

Step 2 downloads the PiC test set from Hugging Face to read the gold answers, so it needs an internet connection the first time. The files that Steps 2 and 3 generate in the same folders (`aggregated_candidates.json`, `aggregated_candidates_deduped.json`, `*_llm_batch.jsonl`) are not committed, the scripts rebuild them in seconds.

To recompute the final scores from the files above, without a GPU or a Gemini account (use `--task pr --config PR-pass` for PR):
```
cd pipeline/
python 01_aggregate_candidates.py --task psd
python 02_dedupe_clusters.py --task psd
python 04_score_predictions.py --task psd
```

The committed n-best and Gemini files are slimmed down with `pipeline/slim_for_release.py` (top 5 spans per question, answer text only). The scripts only read those parts, so the scores are identical.

<!-- LICENSE -->

## License

Distributed under the MIT License. The PiC dataset itself is distributed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).


<!-- CONTACT -->

## Contact

Questions about the code are welcome, email Jannatul Naima at [jannatulnaima.cse.cu@gmail.com](mailto:jannatulnaima.cse.cu@gmail.com) or create a github issue. Thanks!


<!-- ACKNOWLEDGMENTS -->

## Acknowledgements

`retrieval_qa/train/` is adapted from the Hugging Face question answering example (`run_qa.py`) and the PiC authors' changes to it for the PR and PSD tasks (location-aware EM/F1). Please also cite the PiC paper if you use it.

## References

* Thang M. Pham, Seunghyun Yoon, Trung Bui, and Anh Nguyen. 2022. [PiC](https://arxiv.org/abs/2207.09068): A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search. arXiv preprint arXiv:2207.09068.
* Huggingface. 2022. transformers/examples/pytorch/question-answering at main. [https://github.com/huggingface/transformers/tree/main/](https://github.com/huggingface/transformers/tree/main/) examples/pytorch/question-answering.
* Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. [Longformer](https://arxiv.org/abs/2004.05150): The long-document transformer.
* Kevin Clark, Minh-Thang Luong, Quoc V. Le, and Christopher D. Manning. 2020. [ELECTRA](https://arxiv.org/abs/2003.10555): Pre-training text encoders as discriminators rather than generators. In ICLR.
* Yinhan Liu, Myle Ott, Naman Goyal, et al. 2019. [RoBERTa](https://arxiv.org/abs/1907.11692): A robustly optimized BERT pretraining approach.

<p align="right">&#40;<a href="#top">back to top</a>&#41;</p>
