#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
  #"Deehan1866/PR-pass-electra-large"
  #"Deehan1866/longformer-base-4096"
  "Deehan1866/PR-Pass-deberta-v3-base"
  #"Deehan1866/PR-pass-structbert-large"
  #"Deehan1866/PR-pass-xlnet-base-cased"
  #"Deehan1866/PR-pass-roberta-large"
)

for MODEL in "${MODELS[@]}"; do
  # Strip the "Deehan1866/" org prefix to get just the model name
  MODEL_NAME="${MODEL#Deehan1866/}"

  OUTPUT_DIR="./results/phrase_retrieval/PR-pass/qa/Deehan1866/${MODEL_NAME}/predict"
  RUN_NAME="phrase_retrieval/PR-pass/qa/Deehan1866/${MODEL_NAME}/predict"

  echo "=============================================="
  echo "Running prediction for: ${MODEL}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "=============================================="

  python run_qa.py \
    --model_name_or_path "${MODEL}" \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name Deehan1866/phrase_sense_disambiguation \
    --dataset_config_name pass \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --fp16 \
    --gradient_checkpointing \
    --max_seq_length 512 \
    --doc_stride 128 \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_cache \
    --seed 42 \
    --report_to none \
    --run_name "${RUN_NAME}"

  echo "Done: ${MODEL}"
  echo

  # Give the GPU a moment to release memory before the next model in the loop
  sleep 5
done

echo "All predictions complete."