#!/bin/bash

# create data for annotation (Steps 1.1, 1.2, 2.1)
python src/perform_neg_augmentation.py --split train
python src/perform_neg_augmentation.py --split val


# create JNLI-Neg dataset and output to jsonl files
python src/create_jnli_neg.py --split train
python src/create_jnli_neg.py --split val
