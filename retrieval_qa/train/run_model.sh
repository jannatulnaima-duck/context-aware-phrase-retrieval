#!/usr/bin/env bash
# Fine-tunes one QA model on a PiC task, evaluates it, predicts on the test set and
# copies the n-best file to results/ under the name the pipeline expects.
#
# usage (from retrieval_qa/train/):
#   bash run_model.sh <phrase_retrieval|phrase_sense_disambiguation> <PR-pass|PR-page|""> <Longformer|ELECTRA|ROBERTA>
# e.g.
#   bash run_model.sh phrase_retrieval PR-pass Longformer
#   bash run_model.sh phrase_sense_disambiguation "" ELECTRA
set -euo pipefail

DATASET="${1:?task: phrase_retrieval or phrase_sense_disambiguation}"
DATASET_CONFIG="${2-}"          # PR-pass / PR-page for PR, empty for PSD
MODEL="${3:?model: Longformer, ELECTRA or ROBERTA}"

# Hugging Face checkpoints behind the three model names
case "${MODEL}" in
  Longformer) MODEL_ID="allenai/longformer-base-4096" ;;
  ELECTRA)    MODEL_ID="google/electra-base-discriminator" ;;
  ROBERTA)    MODEL_ID="roberta-base" ;;
  *) echo "unknown model '${MODEL}' (use Longformer, ELECTRA or ROBERTA)"; exit 1 ;;
esac

case "${DATASET}" in
  phrase_retrieval)           HF_DATASET="PiC/phrase_retrieval";                    RESULTS_SUBDIR="phrase_retrieval/${DATASET_CONFIG}" ;;
  phrase_sense_disambiguation) HF_DATASET="Deehan1866/phrase_sense_disambiguation"; RESULTS_SUBDIR="phrase_sense_disambiguation" ;;
  *) echo "unknown dataset '${DATASET}'"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QA_DIR="${REPO_ROOT}/results/${RESULTS_SUBDIR}/qa"
OUTPUT_DIR="${QA_DIR}/runs/${MODEL}"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_ARGS=()
if [ -n "${DATASET_CONFIG}" ]; then CONFIG_ARGS=(--dataset_config_name "${DATASET_CONFIG}"); fi

echo "== ${MODEL_ID} on ${HF_DATASET} ${DATASET_CONFIG} -> ${OUTPUT_DIR}"
python "$(dirname "${BASH_SOURCE[0]}")/run_qa.py" \
  --model_name_or_path "${MODEL_ID}" \
  --do_train --do_eval --do_predict \
  --dataset_name "${HF_DATASET}" ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --fp16 \
  --gradient_checkpointing \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir \
  --overwrite_cache \
  --seed 42 \
  --report_to none

# the pipeline looks for exactly this file name
cp "${OUTPUT_DIR}/predict_nbest_predictions.json" "${QA_DIR}/${MODEL}predict_nbest_predictions.json"
echo "n-best file ready: ${QA_DIR}/${MODEL}predict_nbest_predictions.json"
