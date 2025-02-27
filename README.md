# JNLI-Neg

[日本語 | [English](./README_en.md)]

## 概要

否定理解能力を評価するための日本語言語推論データセット **JNLI-Neg** の公開用リポジトリです。
JNLI-Neg は、[JGLUE](https://www.jstage.jst.go.jp/article/jnlp/30/1/30_63/_article/-char/ja)
に含まれる言語推論データセット **JNLI** を拡張して作成しました。

JNLI-Neg データセットは [data/jnli-neg-v1.1/](./data/jnli-neg-v1.1/) ディレクトリに、
JNLI-Neg の作成に使用したソースコードは [src](./src/) ディレクトリに格納しています。

## 目次

- [データセットの統計情報](#データセットの統計情報)
- [データセットの説明](#データセットの説明)
- [ソースコードの説明](#ソースコードの説明)
  - [環境構築](#環境構築)
  - [実行方法](#実行方法)
- [ライセンス](#ライセンス)
- [論文情報](#論文情報)

## データセットの統計情報

|                    | 学習セット | 検証セット |
| ------------------ | ---------: | ---------: |
| NLI インスタンス   |      5,494 |      1,363 |
| 否定のミニマルペア |      6,735 |      1,722 |

「否定のミニマルペア」に関する詳細は、[論文](#論文情報)を参照してください。

## データセットの説明

データセットは JSON 形式です。
各インスタンスは、以下のようなフォーマットで表現されます。

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

key, value (値) の詳細は以下の通りです。

- `id` (integer): JNLI-Neg における id
- `jnli_sentence_pair_id` (integer):
  拡張対象とした JNLI のインスタンスにおける _sentence_pair_id_
- `pair_id_in_group`(integer):
  同一の JNLI インスタンスから拡張して作成した JNLI-Neg インスタンスの集合 (グループ) における id 。
  ある JNLI インスタンス $`I`$ から作成した JNLI-Neg インスタンスの集合
  $`I^{\prime}=\{i^{\prime}_{1}, i^{\prime}_{2}, i^{\prime}_{3}, \dots i^{\prime}_{n} \}`$
  における $`i^{\prime}_{k}`$ の添字 $`k`$ が _pair_id_in_group_ に相当する。
  拡張対象とした JNLI のインスタンスの _pair_id_in_group_ は 0 としている。
- `type` (string):
  前提文と仮説文の組み合わせに関する種類。
  値は以下のいずれか 1 つ。
  - `p_neg`: 前提文にのみ否定要素を含む
  - `h_neg`: 仮説文にのみ否定要素を含む
  - `p_neg_h_neg`: 前提文と仮説文の両方に否定要素を含む
  - `original`: 拡張対象とした JNLI のインスタンス (前提文と仮説文のいずれにも否定要素を含まない)
- `sentence1` (object: 以下で説明): 前提文
- `sentence2` (object: 以下で説明): 仮説文
- `gold_label` (string):
  正解の NLI ラベル。値は以下のいずれか 1 つ。
  - `entailment`: 含意
  - `contradiction`: 矛盾
  - `neutral`: 中立
- `annotator_labels` (object):
  3 人の作業者による人手アノテーションの結果

`sentence1` 及び `sentence2` の値のフォーマットは以下の通りです。

- `sentence` (string): 平文
- `neg` (object | null):
  文に含まれる否定の情報。
  否定を含まない文の値は null 。
  否定を含む文における値は以下の形式。
  - `neg_id_in_jnli_sentence` (integer):
    1 つの JNLI の文から作成した否定を含む文の集合における id 。
    ある JNLI の文 (前提文と仮説文のいずれか) $s$ から作成した 否定を含む文の集合
    $`S^{\prime}=\{s^{\prime}_{1}, s^{\prime}_{2}, s^{\prime}_{3}, \dots s^{\prime}_{n} \}`$
    における $`s^{\prime}_{k}`$ の添字 $`k`$ が _neg_id_in_jnli_sentence_ に相当する。
  - `neg_position` (string): 否定要素の位置。値は以下のいずれか 1 つ。
    - `end`: 文末
    - `mid`: 文の途中
  - `target_pos` (string): 拡張において、否定要素を付与した形態素の品詞。
    値は `動詞`, `形容詞`, `形容動詞` のいずれか 1 つ。

## ソースコードの説明

JNLI-Neg の構築に利用したソースコードを公開しています。
ソースコードを実行することで、以下の動作を再現できます。

- JNLI のデータを読み込み、新たな文の組 $(p, h)$ を作成する
  (論文の拡張手法における手順 1.1, 1.2, 及び 2.1)
- 人手アノテーション (手順 2.2) の結果を読み込み、JNLI-Neg データセットを作成する

### 環境構築

このリポジトリを実行環境に保存 (fork + clone など) した上で、
以下の手順で環境構築を行ってください。
python 3.10 以降を実行できる環境を使用してください。

1. JNLI データセットのダウンロード

   [JGLUE のリポジトリにおける JNLI 格納ディレクトリ](https://github.com/yahoojapan/JGLUE/tree/main/datasets/jnli-v1.1) から
   `train-v1.1.json` 及び `valid-v1.1.json` をダウンロードし、
   本リポジトリの [data/jnli-v1.1](./data/jnli-v1.1/) ディレクトリ直下に格納してください。

2. MeCab 及び UniDic の準備

   本ソースコードでは、形態素解析器 [MeCab](https://taku910.github.io/mecab/) 及び形態素解析辞書 [UniDic](https://clrd.ninjal.ac.jp/unidic/) を使用します。
   [MeCab の公式ページ](https://taku910.github.io/mecab/) などを参照して、実行環境に MeCab 及び UniDic をインストールしてください。
   さらに、MeCab の解析用辞書を UniDic に設定してください。
   UniDic 以外の辞書 (IPA など) を利用する設定になっている場合、ソースコードが正しく動作しないので注意してください。

3. 必要な python パッケージのインストール

   [requirements.txt](./requirements.txt) を参照して、
   必要な python パッケージを実行環境にインストールしてください。
   pip を使用する場合、以下のコマンドで一括インストールできます。

   ```sh
   pip install -r requirements.txt
   ```

### 実行方法

本リポジトリのルートディレクトリで [run.sh](run.sh) を実行することにより、処理を実行することができます。
python 3.10 以降のバージョンを利用してください。

## ライセンス

JNLI-Neg データセットとソースコードは、それぞれ以下のライセンスで配布します。

- JNLI-Neg データセット: [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- ソースコード: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## 論文情報

本リポジトリを利用する場合、以下の論文を引用してください。

```bibtex
@inproceedings{yoshida-et-al-anlp2025,
  title={否定理解能力を評価するための日本語言語推論データセットの構築},
  author={吉田朝飛 and 加藤芳秀 and 小川泰弘 and 松原茂樹},
  booktitle={言語処理学会第31回年次大会発表論文集},
  year={2025},
}
```
