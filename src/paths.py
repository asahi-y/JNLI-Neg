import literals as L

# JNLI dataset
jnli_dataset: dict[L.SPLIT, str] = {
    "train": "data/jnli-v1.1/train-v1.1.json",
    "val": "data/jnli-v1.1/valid-v1.1.json",
}

# llm filtering (Step 1.2) output logs
llm_filtering_output_files = {
    "train": "logs/llm_filtering_output/concat/llm_output-train.jsonl",
    "val": "logs/llm_filtering_output/concat/llm_output-valid.jsonl",
}

# negation cue detection output logs
cue_detection_output_files = {
    "train": "logs/jnli_neg_instance/neg_instances-train.jsonl",
    "val": "logs/jnli_neg_instance/neg_instances-valid.jsonl",
}

# new sentence pairs for manual annotation
annotation_target_files: dict[L.SPLIT, str] = {
    "train": f"annotations/jnli-neg-annotation-targets/train-sampling.jsonl",
    "val": f"annotations/jnli-neg-annotation-targets/valid-sampling.jsonl",
}

# JNLI-Neg annotation results
annotation_results_dir: dict[L.SPLIT, str] = {
    "train": "annotations/jnli-neg-annotation-results/cleaned/train/",
    "val": "annotations/jnli-neg-annotation-results/cleaned/valid/",
}

# JNLI-Neg output path
jnli_neg_dataset: dict[L.SPLIT, str] = {
    "train": "data/jnli-neg-v1.1/jnli-neg-train-v1.1.jsonl",
    "val": "data/jnli-neg-v1.1/jnli-neg-valid-v1.1.jsonl",
}
