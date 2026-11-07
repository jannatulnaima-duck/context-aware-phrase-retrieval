# small helpers that all four pipeline scripts share
import os

from datasets import load_dataset

# PSD points at the copy of the dataset we trained on. PR defaults to the
# public PiC release, change it with --hf_dataset if yours lives elsewhere.
HF_DATASETS = {
    "psd": "Deehan1866/phrase_sense_disambiguation",
    "pr": "PiC/phrase_retrieval",
}


def add_task_args(parser):
    parser.add_argument("--task", choices=["psd", "pr"], required=True,
                        help="which PiC task to run the pipeline on")
    parser.add_argument("--config", default="PR-pass",
                        help="PR only: PR-pass or PR-page (ignored for psd)")
    parser.add_argument("--hf_dataset", default=None,
                        help="override the HF dataset name")
    parser.add_argument("--results_root", default="../results")


def task_dir(args):
    # one results folder per task (and per config for PR)
    if args.task == "psd":
        return os.path.join(args.results_root, "phrase_sense_disambiguation")
    return os.path.join(args.results_root, "phrase_retrieval", args.config)


def load_test_split(args):
    name = args.hf_dataset or HF_DATASETS[args.task]
    if args.task == "pr":
        return load_dataset(name, args.config, split="test")
    return load_dataset(name, split="test")
