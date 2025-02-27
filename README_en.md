# JNLI-Neg

[[日本語](./README.md) | English]

## Abstract

This repository provides **JNLI-Neg**, a Japanese natural language inference dataset for evaluating negation understanding capability.
JNLI-Neg is an extended version of **JNLI**, a Japanese natural language inference dataset that is included in [JGLUE](https://aclanthology.org/2022.lrec-1.317/).

The JNLI-Neg dataset is stored in the [data/jnli-neg-v1.1/](./data/jnli-neg-v1.1/) directory, and the source code used to construct JNLI-Neg is stored in the [src](./src/) directory.

## Table of Contents

- [Dataset Statistics](#dataset-statistics)
- [Dataset Description](#dataset-description)
- [Source Code Description](#source-code-description)
  - [Setup](#setup)
  - [Run](#run)
- [License](#license)
- [Paper](#paper)

## Dataset Statistics

|                          | Training | Validation |
| ------------------------ | -------: | ---------: |
| NLI Instance             |    5,494 |      1,363 |
| Minimal Pair of Negation |    6,735 |      1,722 |

For the detail of "minimal pair of negation", please refer to [our paper](#paper).

## Dataset Description

JNLI-Neg is provided in the JSON format,
with each instance structured as shown below.

```json
{
  "id": 236,
  "jnli_sentence_pair_id": 1002,
  "pair_id_in_group": 1,
  "type": "p_neg",
  "sentence1": {
    "sentence": "街中を黄色くないトラックが走っています。",
    "neg": {
      "neg_id_in_jnli_sentence": 0,
      "neg_position": "mid",
      "target_pos": "形容詞"
    }
  },
  "sentence2": {
    "sentence": "街中を赤いトラックが走っています。",
    "neg": null
  },
  "gold_label": "neutral",
  "annotator_labels": {
    "entailment": 0,
    "neutral": 3,
    "contradiction": 0
  }
}
```

The details of the keys and values are as follows:

- `id` (integer): instance id in JNLI-Neg
- `jnli_sentence_pair_id` (integer):
  _sentence_pair_id_ of the corresponding instance in JNLI that was used for augmentation
- `pair_id_in_group`(integer):
  id within the group of JNLI-Neg instances derived from the same JNLI instance.
  If a JNLI instance $`I`$ is augmented into a set of JNLI-Neg instances
  $`I^{\prime}=\{i^{\prime}_{1}, i^{\prime}_{2}, i^{\prime}_{3}, \dots, i^{\prime}_{n} \}`$, then the index $`k`$ of $`i^{\prime}_{k}`$ corresponds to _pair_id_in_group_.
  The original JNLI instance used for augmentation has _pair_id_in_group_ set to 0.
- `type` (string):
  type of the premise-hypothesis pair based on negation presence.
  Possible values:
  - `p_neg`: only the premise contains negation
  - `h_neg`: only the hypothesis contains negation
  - `p_neg_h_neg`: both the premise and hypothesis contain negation
  - `original`: original JNLI instance (neither the premise nor hypothesis contains negation)
- `sentence1` (object: explained below): premise sentence
- `sentence2` (object: explained below): hypothesis sentence
- `gold_label` (string): ground-truth NLI label, with possible values:
  - `entailment`
  - `contradiction`
  - `neutral`
- `annotator_labels` (object):
  annotation results provided by three human annotators

The format of the values for `sentence1` and `sentence2` is as follows:

- `sentence` (string): plain text of the sentence
- `neg` (object | null):
  information about negation, if present in the sentence.
  If the sentence does not contain negation, the value is null.
  If the sentence contains negation, the value is the object with the following fields:
  - `neg_id_in_jnli_sentence` (integer):
    id within the set of sentences derived from a single JNLI sentence.
    Given a JNLI sentence $`s`$ (either a premise or hypothesis) augmented into a set
    $`S^{\prime}=\{s^{\prime}_{1}, s^{\prime}_{2}, s^{\prime}_{3}, \dots, s^{\prime}_{n} \}`$, the index $`k`$ of $`s^{\prime}_{k}`$ corresponds to _neg_id_in_jnli_sentence_.
  - `neg_position` (string): position of the negation cue in the sentence.
    Possible values:
    - `end`: end of the sentence
    - `mid`: middle of the sentence
  - `target_pos` (string): part of speech of the morpheme to which the negation cue was added during augmentation.
    Possible values:
    - `動詞` (verb)
    - `形容詞` (adjective)
    - `形容動詞` (adjectival verb)

## Source Code Description

We have released the source code used to construct JNLI-Neg.
By running the source code, you can reproduce the following process:

- Load the JNLI dataset and generate new sentence pairs $(p, h)$
  (corresponding to Steps 1.1, 1.2, and 2.1 in our paper).
- Load the results of manual annotation (Step 2.2) and generate the JNLI-Neg dataset.

### Setup

First, save this repository to your execution environment (e.g., fork and clone).
Then, set up the environment by following the steps below.
Please use an environment that supports Python 3.10 or later.

1. Download JNLI dataset

   Download `train-v1.1.json` and `valid-v1.1.json` from [the JNLI directory in JGLUE](https://github.com/yahoojapan/JGLUE/tree/main/datasets/jnli-v1.1), and place them in the [data/jnli-v1.1](./data/jnli-v1.1) directory.

2. Set up MeCab and UniDic

   This source code uses the morphological analyzer [MeCab](https://taku910.github.io/mecab/) and the morphological dictionary [UniDic](https://clrd.ninjal.ac.jp/unidic/).
   Install MeCab and UniDic by referring to
   [MeCab's official page](https://taku910.github.io/mecab/) and other resources.
   In addition, configure MeCab to use UniDic as its dictionary.
   If MeCab is set to use a different dictionary (e.g., IPA), the code may not work correctly.

3. Install required Python packages

   Install the required Python packages listed in [requirements.txt](./requirements.txt).
   If you are using pip, you can run:

   ```sh
   pip install -r requirements.txt
   ```

### Run

Execute [run.sh](./run.sh) in the root directory.
Make sure to use Python 3.10 or later.

## License

The JNLI-Neg dataset and source code are distributed under the following licenses:

- JNLI-Neg dataset: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- source code: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Paper

If you use this repository in your work, please reference the following paper.

```bibtex
@inproceedings{yoshida-et-al-anlp2025,
  title={否定理解能力を評価するための日本語言語推論データセットの構築},
  author={吉田朝飛 and 加藤芳秀 and 小川泰弘 and 松原茂樹},
  booktitle={言語処理学会第31回年次大会発表論文集},
  year={2025},
  note={in Japanese},
}
```
