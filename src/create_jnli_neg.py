import os
import literals as L
import paths as P
import utils
from typing import get_args, Literal
from tap import Tap


class AugmentedSentence:
    """
    negation augmented sentence
    """

    def __init__(
        self, p_or_h: L.NLI_P_OR_H, sentence: str, aug: "NegAugmentation | None"
    ) -> None:
        self.p_or_h = p_or_h
        self.sentence = sentence
        self.aug = aug


class NegAugmentation:
    """
    augmentation information
    """

    def __init__(
        self,
        aug_id: int,
        neg_position: L.AUGMENTED_NEG_POSITION,
        target_pos: L.AUG_TARGET_POS,
    ) -> None:
        self.aug_id = aug_id
        self.neg_position: L.AUGMENTED_NEG_POSITION = neg_position
        self.target_pos: L.AUG_TARGET_POS = target_pos


class AnnotatedLabels:
    def __init__(self, labels: dict[L.ANNOTATOR, L.NLI_LABEL]) -> None:
        self.label1: L.NLI_LABEL = labels["annotator01"]
        self.label2: L.NLI_LABEL = labels["annotator02"]
        self.label3: L.NLI_LABEL = labels["annotator03"]
        nli_labels = get_args(L.NLI_LABEL)
        assert (
            self.label1 in nli_labels
            and self.label2 in nli_labels
            and self.label3 in nli_labels
        )

    def get_consensus_label(self, consensus_count: int = 2) -> L.NLI_LABEL | None:
        """
        Returns the consensus label if 2 or more annotators agree, otherwise None.
        """

        label_counts = self.get_count_by_label()

        for label, count in label_counts.items():
            if count >= consensus_count:
                return label
        return None

    def get_count_by_label(self) -> dict[L.NLI_LABEL, int]:
        label_counts = {label: 0 for label in get_args(L.NLI_LABEL)}
        label_counts[self.label1] += 1
        label_counts[self.label2] += 1
        label_counts[self.label3] += 1

        return label_counts


class NegNLIInstance:
    """
    NLI instance that contains negation
    (elements of D_{neg})
    """

    def __init__(
        self,
        # ids assigned to randomly sampled data
        sampled_id: int,
        original_sentence_pair_id: int,
        new_pair_id_in_sentence: int,
        type: L.NLI_AUG_COMB_PATTERN,
        sentence1: AugmentedSentence,
        sentence2: AugmentedSentence,
        labels: AnnotatedLabels,
    ) -> None:
        self.sampled_id = sampled_id
        self.original_sentence_pair_id = original_sentence_pair_id
        self.new_pair_id_in_sentence = new_pair_id_in_sentence
        self.type: L.NLI_AUG_COMB_PATTERN = type
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels


class AugmentedDataset:
    """
    Dataset consisting of augmented NLI instances (D_{neg})
    """

    def __init__(self, split: L.SPLIT, jsonl_dir: str) -> None:
        self.split = split
        self.jsonl_dir = jsonl_dir.rstrip("/")
        self.instances: list[NegNLIInstance] = []

        # load jsonl files
        for file in sorted(os.listdir(jsonl_dir)):
            if file.endswith(".jsonl"):
                self.instances.extend(self._load_instances(f"{jsonl_dir}/{file}"))

    def _load_instances(self, jsonl_file: str) -> list[NegNLIInstance]:
        return_list: list[NegNLIInstance] = []
        loaded_jsonl = utils.load_jsonl(jsonl_file)
        for json in loaded_jsonl:
            neg_nli_instance = NegNLIInstance(
                sampled_id=json["id"],
                original_sentence_pair_id=json["original_sentence_pair_id"],
                new_pair_id_in_sentence=json["new_pair_id_in_sentence"],
                type=json["type"],
                sentence1=AugmentedSentence(
                    p_or_h="premise",
                    sentence=json["sentence1"]["sentence"],
                    aug=(
                        None
                        if json["sentence1"]["aug"] is None
                        else NegAugmentation(
                            aug_id=json["sentence1"]["aug"]["aug_id"],
                            neg_position=json["sentence1"]["aug"]["neg_position"],
                            target_pos=json["sentence1"]["aug"]["target_pos"],
                        )
                    ),
                ),  # type: ignore
                sentence2=AugmentedSentence(
                    p_or_h="hypothesis",
                    sentence=json["sentence2"]["sentence"],
                    aug=(
                        None
                        if json["sentence2"]["aug"] is None
                        else NegAugmentation(
                            aug_id=json["sentence2"]["aug"]["aug_id"],
                            neg_position=json["sentence2"]["aug"]["neg_position"],
                            target_pos=json["sentence2"]["aug"]["target_pos"],
                        )
                    ),
                ),  # type: ignore
                labels=AnnotatedLabels(labels=json["labels"]),
            )
            return_list.append(neg_nli_instance)

        return return_list


class JNLIInstance:
    """
    JNLI instance (elements of D_{JNLI})
    """

    def __init__(
        self,
        sentence_pair_id: int,  # str -> int
        yjcaptions_id: str,
        sentence1: str,
        sentence2: str,
        label: L.NLI_LABEL,
    ) -> None:
        self.sentence_pair_id = sentence_pair_id
        self.yjcaptions_id = yjcaptions_id
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label: L.NLI_LABEL = label


class JNLIDataset:
    """
    JNLI dataset (D_{JNLI})
    """

    def __init__(self, split: L.SPLIT, jsonl_file: str) -> None:
        self.split = split
        self.jsonl_file = jsonl_file
        self.instances: list[JNLIInstance] = self._load_instances(jsonl_file)

    def _load_instances(self, jsonl_file: str) -> list[JNLIInstance]:
        return_list: list[JNLIInstance] = []
        loaded_jsonl = utils.load_jsonl(jsonl_file)
        for json in loaded_jsonl:
            nli = JNLIInstance(
                sentence_pair_id=int(json["sentence_pair_id"]),
                yjcaptions_id=json["yjcaptions_id"],
                sentence1=json["sentence1"],
                sentence2=json["sentence2"],
                label=json["label"],
            )
            return_list.append(nli)

        return return_list

    def __getitem__(self, sentence_paid_id: int) -> JNLIInstance:
        for instance in self.instances:
            if instance.sentence_pair_id == sentence_paid_id:
                return instance
        raise ValueError(f"Instance with sentence_pair_id {sentence_paid_id} not found")


class JNLINegDataset:
    """
    JNLI-Neg dataset (D_{JNLI-Neg})
    """

    def __init__(
        self,
        split: L.SPLIT,
        neg_augmented_jsonl_dir: str,
        jnli_jsonl_file: str,
    ) -> None:
        self.split = split
        # augmented negation dataset (D_{neg})
        self.augmented_neg_dataset = AugmentedDataset(split, neg_augmented_jsonl_dir)
        # JNLI dataset (D_{JNLI})
        self.jnli_dataset = JNLIDataset(split, jnli_jsonl_file)

        # JNLI-Neg instances (elements of D_{JNLI-Neg})
        self.instances: list[JNLINegInstance] = self._create_dataset()

        # minimal pairs in JNLI-Neg (M)
        self.minimal_pairs = self._create_minimal_pairs()

    def _create_minimal_pairs(self) -> list["JNLINegMinimalPair"]:
        minimal_pairs: list["JNLINegMinimalPair"] = []

        jnli_sentence_pair_ids = set(
            [instance.jnli_sentence_pair_id for instance in self.instances]
        )

        for jnli_sentence_pair_id in jnli_sentence_pair_ids:
            # JNLI instance by jnli_sentence_pair_id
            instances_by_jnli = [
                instance
                for instance in self.instances
                if instance.jnli_sentence_pair_id == jnli_sentence_pair_id
            ]

            # create minimal pairs (elements of M_{p}, M_{h})
            original_instance = [
                instance
                for instance in instances_by_jnli
                if instance.type == "original"
            ]
            assert len(original_instance) == 1
            for instance in [
                instance
                for instance in instances_by_jnli
                if instance.type in ["p_neg", "h_neg"]
            ]:
                minimal_pair_check = JNLINegMinimalPair.check_minimal_pair(
                    original_instance[0], instance
                )
                assert minimal_pair_check != "REVERSE"
                if minimal_pair_check is True:
                    minimal_pairs.append(
                        JNLINegMinimalPair(
                            original_instance[0], instance, jnli_sentence_pair_id
                        )
                    )

            # create minimal pairs (elements of M_{p, ph}, M_{h, ph})
            p_heg_h_neg_instances = [
                instance
                for instance in instances_by_jnli
                if instance.type == "p_neg_h_neg"
            ]
            for p_neg_h_neg_instance in p_heg_h_neg_instances:
                for instance in [
                    instance
                    for instance in instances_by_jnli
                    if instance.type in ["p_neg", "h_neg"]
                ]:
                    minimal_pair_check = JNLINegMinimalPair.check_minimal_pair(
                        instance, p_neg_h_neg_instance
                    )
                    assert minimal_pair_check != "REVERSE"
                    if minimal_pair_check is True:
                        minimal_pairs.append(
                            JNLINegMinimalPair(
                                instance, p_neg_h_neg_instance, jnli_sentence_pair_id
                            )
                        )

        return minimal_pairs

    def create_jsonl(self, output_file_path: str) -> None:
        output_list = []
        for instance in self.instances:
            output_list.append(
                {
                    "id": instance.id,
                    "jnli_sentence_pair_id": instance.jnli_sentence_pair_id,
                    "pair_id_in_group": instance.pair_id_in_group,
                    "type": instance.type,
                    "sentence1": {
                        "sentence": instance.sentence1.sentence,
                        "neg": (
                            None
                            if instance.sentence1.neg is None
                            else {
                                "neg_id_in_jnli_sentence": instance.sentence1.neg.neg_id_in_jnli_sentence,
                                "neg_position": instance.sentence1.neg.neg_position,
                                "target_pos": instance.sentence1.neg.target_pos,
                            }
                        ),
                    },
                    "sentence2": {
                        "sentence": instance.sentence2.sentence,
                        "neg": (
                            None
                            if instance.sentence2.neg is None
                            else {
                                "neg_id_in_jnli_sentence": instance.sentence2.neg.neg_id_in_jnli_sentence,
                                "neg_position": instance.sentence2.neg.neg_position,
                                "target_pos": instance.sentence2.neg.target_pos,
                            }
                        ),
                    },
                    "gold_label": instance.gold_label,
                    "annotator_labels": instance.annotator_labels,
                }
            )

        utils.save_jsonl(output_file_path, output_list)
        print(f"\nJNLI-Neg dataset was saved to {output_file_path}")
        print()

    def _create_dataset(self) -> list["JNLINegInstance"]:
        return_list: list["JNLINegInstance"] = []

        previous_jnli_sentence_pair_id: int = -1
        id: int = 0
        pair_id_in_group: int = 0
        for neg_instance in self.augmented_neg_dataset.instances:
            # Original Sentence Pair
            if neg_instance.original_sentence_pair_id != previous_jnli_sentence_pair_id:
                pair_id_in_group = 0
                jnli_instance = self.jnli_dataset[
                    neg_instance.original_sentence_pair_id
                ]
                sentence1 = NegSentence(sentence=jnli_instance.sentence1, neg=None)
                sentence2 = NegSentence(sentence=jnli_instance.sentence2, neg=None)
                new_instance_by_jnli = JNLINegInstance(
                    id=id,
                    jnli_sentence_pair_id=jnli_instance.sentence_pair_id,
                    pair_id_in_group=pair_id_in_group,
                    type="original",
                    sentence1=sentence1,
                    sentence2=sentence2,
                    gold_label=jnli_instance.label,
                    annotator_labels=None,
                )
                return_list.append(new_instance_by_jnli)
                id += 1
                pair_id_in_group += 1

            # Augmented Sentence Pair
            # discard instances without consensus label
            if neg_instance.labels.get_consensus_label() is None:
                previous_jnli_sentence_pair_id = neg_instance.original_sentence_pair_id
                continue

            sentence1 = NegSentence(
                sentence=neg_instance.sentence1.sentence,
                neg=(
                    None
                    if neg_instance.sentence1.aug is None
                    else JNLINegAugmentedNegation(
                        neg_id_in_jnli_sentence=neg_instance.sentence1.aug.aug_id,
                        neg_position=(
                            "end"
                            if neg_instance.sentence1.aug.neg_position == "clausal"
                            else "mid"
                        ),
                        target_pos=neg_instance.sentence1.aug.target_pos,
                    )
                ),
            )
            sentence2 = NegSentence(
                sentence=neg_instance.sentence2.sentence,
                neg=(
                    None
                    if neg_instance.sentence2.aug is None
                    else JNLINegAugmentedNegation(
                        neg_id_in_jnli_sentence=neg_instance.sentence2.aug.aug_id,
                        neg_position=(
                            "end"
                            if neg_instance.sentence2.aug.neg_position == "clausal"
                            else "mid"
                        ),
                        target_pos=neg_instance.sentence2.aug.target_pos,
                    )
                ),
            )
            consensus_label = neg_instance.labels.get_consensus_label()
            assert consensus_label is not None
            new_instance_neg = JNLINegInstance(
                id=id,
                jnli_sentence_pair_id=neg_instance.original_sentence_pair_id,
                pair_id_in_group=pair_id_in_group,
                type=neg_instance.type,
                sentence1=sentence1,
                sentence2=sentence2,
                gold_label=consensus_label,
                annotator_labels=neg_instance.labels.get_count_by_label(),
            )
            return_list.append(new_instance_neg)

            id += 1
            pair_id_in_group += 1
            previous_jnli_sentence_pair_id = neg_instance.original_sentence_pair_id

        return return_list


class JNLINegInstance:
    """
    JNLI-Neg instance (element of D_{JNLI-Neg})
    """

    def __init__(
        self,
        id: int,
        jnli_sentence_pair_id: int,
        pair_id_in_group: int,  # group id (grouped by JNLI instance)
        type: (
            L.NLI_AUG_COMB_PATTERN | L.JNLI_ORIGINAL
        ),  # "original", "p_neg", "h_neg", "p_neg_h_neg"
        sentence1: "NegSentence",
        sentence2: "NegSentence",
        gold_label: L.NLI_LABEL,
        annotator_labels: dict[L.NLI_LABEL, int] | None,
    ) -> None:
        self.id = id
        self.jnli_sentence_pair_id = jnli_sentence_pair_id
        self.pair_id_in_group = pair_id_in_group
        self.type: L.NLI_AUG_COMB_PATTERN | L.JNLI_ORIGINAL = type
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.gold_label: L.NLI_LABEL = gold_label
        self.annotator_labels: dict[L.NLI_LABEL, int] | None = annotator_labels


class NegSentence:
    """
    sentence that contains negation
    """

    def __init__(
        self,
        sentence: str,  # plain_sentence
        neg: "JNLINegAugmentedNegation | None",  # negation info
    ) -> None:
        self.sentence = sentence
        self.neg = neg


class JNLINegAugmentedNegation:
    """
    negation info in augmented sentence
    """

    def __init__(
        self,
        neg_id_in_jnli_sentence: int,
        neg_position: L.AUGMENTED_NEG_POSITION_JNLI_NEG,  # "end", "mid"
        target_pos: L.AUG_TARGET_POS,  # "動詞", "形容詞", "形容動詞"
    ) -> None:
        self.neg_id_in_jnli_sentence = neg_id_in_jnli_sentence
        self.neg_position: L.AUGMENTED_NEG_POSITION_JNLI_NEG = neg_position
        self.target_pos: L.AUG_TARGET_POS = target_pos


class JNLINegMinimalPair:
    """
    minimal pair of NLI instance
    """

    def __init__(
        self,
        instance1: JNLINegInstance,
        instance2: JNLINegInstance,
        jnli_sentence_pair_id: int,
    ) -> None:
        self.instance1 = instance1
        self.instance2 = instance2
        self.jnli_sentence_pair_id = jnli_sentence_pair_id

        self.type: L.MINIMAL_PAIR_TYPE

        # important negation or not
        self.is_important_negation: bool = (
            True if self.instance1.gold_label != self.instance2.gold_label else False
        )

        if self.instance1.jnli_sentence_pair_id != self.instance2.jnli_sentence_pair_id:
            raise ValueError(
                f"Instances are not from the same JNLI sentence pair: ({self.instance1.jnli_sentence_pair_id}, {self.instance2.jnli_sentence_pair_id})"
            )

        # judge minimal pair type
        if self.instance1.type == "original" and self.instance2.type == "p_neg":
            self.type = "PH-PnegH"
        elif self.instance1.type == "original" and self.instance2.type == "h_neg":
            self.type = "PH-PHneg"
        elif (
            self.instance1.type == "p_neg"
            and self.instance2.type == "p_neg_h_neg"
            and self.instance1.sentence1.neg
            and self.instance2.sentence1.neg
            and self.instance1.sentence1.neg.neg_id_in_jnli_sentence
            == self.instance2.sentence1.neg.neg_id_in_jnli_sentence
        ):
            self.type = "PnegH-PnegHneg"
        elif (
            self.instance1.type == "h_neg"
            and self.instance2.type == "p_neg_h_neg"
            and self.instance1.sentence2.neg
            and self.instance2.sentence2.neg
            and self.instance1.sentence2.neg.neg_id_in_jnli_sentence
            == self.instance2.sentence2.neg.neg_id_in_jnli_sentence
        ):
            self.type = "PHneg-PnegHneg"
        else:
            raise ValueError(
                f"The pair is not a minimal pair: ({self.instance1.type}, {self.instance2.type})"
            )

    @classmethod
    def check_minimal_pair(
        cls, instance1: JNLINegInstance, instance2: JNLINegInstance
    ) -> bool | Literal["REVERSE"]:
        """
        judge if the instance pair is a minimal pair or not
        - True: the pair is a minimal pair
        - REVERSE: the pair is a minimal pair but the order is reversed
        - False: the pair is not a minimal pair
        """
        if instance1.jnli_sentence_pair_id != instance2.jnli_sentence_pair_id:
            return False
        # PH-PnegH
        elif instance1.type == "original" and instance2.type == "p_neg":
            return True
        # PH-PHneg
        elif instance1.type == "original" and instance2.type == "h_neg":
            return True
        # PnegH-PnegHneg
        elif (
            instance1.type == "p_neg"
            and instance2.type == "p_neg_h_neg"
            and instance1.sentence1.neg
            and instance2.sentence1.neg
            and instance1.sentence1.neg.neg_id_in_jnli_sentence
            == instance2.sentence1.neg.neg_id_in_jnli_sentence
        ):
            return True
        # PHneg-PnegHneg
        elif (
            instance1.type == "h_neg"
            and instance2.type == "p_neg_h_neg"
            and instance1.sentence2.neg
            and instance2.sentence2.neg
            and instance1.sentence2.neg.neg_id_in_jnli_sentence
            == instance2.sentence2.neg.neg_id_in_jnli_sentence
        ):
            return True
        # REVERSE
        elif instance2.type == "original" and instance1.type == "p_neg":
            return "REVERSE"
        elif instance2.type == "original" and instance1.type == "h_neg":
            return "REVERSE"
        elif (
            instance2.type == "p_neg"
            and instance1.type == "p_neg_h_neg"
            and instance2.sentence1.neg
            and instance1.sentence1.neg
            and instance2.sentence1.neg.neg_id_in_jnli_sentence
            == instance1.sentence1.neg.neg_id_in_jnli_sentence
        ):
            return "REVERSE"
        elif (
            instance2.type == "h_neg"
            and instance1.type == "p_neg_h_neg"
            and instance2.sentence2.neg
            and instance1.sentence2.neg
            and instance2.sentence2.neg.neg_id_in_jnli_sentence
            == instance1.sentence2.neg.neg_id_in_jnli_sentence
        ):
            return "REVERSE"
        else:
            return False


class Args(Tap):
    split: L.SPLIT


if __name__ == "__main__":
    args = Args().parse_args()
    jnli_neg_dataset = JNLINegDataset(
        split=args.split,
        neg_augmented_jsonl_dir=P.annotation_results_dir[args.split],
        jnli_jsonl_file=P.jnli_dataset[args.split],
    )
    jnli_neg_dataset.create_jsonl(P.jnli_neg_dataset[args.split])
