from pathlib import Path
from typing import Literal, get_args
from tqdm import tqdm
from morph_analysis import MorphAnalysis
from neg_augmentation import NegationAugmentation
from consts import LLM_OUTPUT_CORRECT, LLM_OUTPUT_INCORRECT, AugSentenceType
from tap import Tap
from openai import OpenAI
import literals as L
import configs as cfg
import os
import utils
import paths
import random


class JNLIDataset:

    def __init__(
        self,
        file_path: str,
        split: L.SPLIT,
        dataset_name: str = "jnli",
    ) -> None:

        # set to True after Step 1.1
        self.is_augmented: bool = False

        # set to True after Step 1.2
        self.is_llm_filtered: bool = False

        loaded_json = utils.load_jsonl(file_path)

        self.split = split
        self.dataset_name = dataset_name
        self.nli_instances: list[NLIInstance] = []

        print(f"\nLoading JNLI {self.split} dataset...")
        for nli_instance in tqdm(loaded_json):
            nli_instance = NLIInstance(
                nli_instance["sentence_pair_id"],
                nli_instance["yjcaptions_id"],
                nli_instance["sentence1"],
                nli_instance["sentence2"],
                nli_instance["label"],
            )

            self.nli_instances.append(nli_instance)

        self.aug_stat = AugmentationStatistics(
            original_nli_pairs=len(self.nli_instances)
        )

        print(f"\nLoaded {len(self.nli_instances)} NLI instances.\n")

    def __len__(self) -> int:
        return len(self.nli_instances)

    def get_item_by_sentence_pair_id(self, sentence_pair_id: int) -> "NLIInstance":
        for nli_instance in self.nli_instances:
            if nli_instance.sentence_pair_id == sentence_pair_id:
                return nli_instance
        raise ValueError(f"Invalid sentence_pair_id: {sentence_pair_id}")

    def _set_is_augmented(self) -> None:
        if self.is_augmented:
            raise ValueError("Augmentation has already been performed.")
        else:
            self.is_augmented = True

    def _set_is_llm_filtered(self) -> None:
        if self.is_llm_filtered:
            raise ValueError("LLM filtering has already been performed.")
        else:
            self.is_llm_filtered = True

    def perform_augmentation(self) -> None:
        """
        Step 1.1
        create sentences that contain negation cues
        """
        for nli_instance in self.nli_instances:
            nli_instance.perform_augmentation_for_jnli_instance()

            # set augmentation info
            # premise
            for aug in nli_instance.sentence1.neg_augmentation.augmented_negs:
                assert not aug.neg_position is None
                self.aug_stat.update_augmentation_results(
                    "premise", aug.neg_position, aug.target_pos_simple
                )
            # hypothesis
            for aug in nli_instance.sentence2.neg_augmentation.augmented_negs:
                assert not aug.neg_position is None
                self.aug_stat.update_augmentation_results(
                    "hypothesis", aug.neg_position, aug.target_pos_simple
                )

        self._set_is_augmented()

    def check_sentence_naturalness(
        self,
        request_api: bool,
        log_output_file_path_name: str,
        start_idx: int = 0,
        end_idx: int = -1,
    ) -> None:
        """
        Step 1.2
        check the naturalness of the sentences using LLM (OpenAI API)
        - If request_api is True, send a request to the API
        - If request_api is False, read the LLM output from the log
            (this can be used when log files are already created)
        """
        if request_api:
            # request to the API
            self._check_sentence_naturalness_by_openai_api(
                log_output_file_path_name=log_output_file_path_name,
                start_idx=start_idx,
                end_idx=end_idx,
            )
        else:
            # load a log file
            loaded_json = utils.load_jsonl(log_output_file_path_name)

            print(f"\nLoading llm output from log from {log_output_file_path_name} ...")
            for nli_instance in tqdm(self.nli_instances):
                for log in loaded_json:
                    if nli_instance.sentence_pair_id == log["sentence_pair_id"]:
                        for (
                            aug
                        ) in nli_instance.sentence1.neg_augmentation.augmented_negs:
                            for log_aug in log["sentence1"]["augmented"]:
                                if aug.id_in_sent == log_aug["id_in_sent"]:
                                    assert (
                                        aug.augmented_sent(nli_instance.sentence1.nodes)
                                        == log_aug["augmented_sent"]
                                    )
                                    aug.set_llm_output(log_aug["llm_output"])
                        for (
                            aug
                        ) in nli_instance.sentence2.neg_augmentation.augmented_negs:
                            for log_aug in log["sentence2"]["augmented"]:
                                if aug.id_in_sent == log_aug["id_in_sent"]:
                                    assert (
                                        aug.augmented_sent(nli_instance.sentence2.nodes)
                                        == log_aug["augmented_sent"]
                                    )
                                    aug.set_llm_output(log_aug["llm_output"])
                        break

                for aug in nli_instance.sentence1.neg_augmentation.augmented_negs:
                    if aug.llm_output is None:
                        raise ValueError(
                            f"LLM output is not set for sentence pair {nli_instance.sentence_pair_id}"
                        )
                for aug in nli_instance.sentence2.neg_augmentation.augmented_negs:
                    if aug.llm_output is None:
                        raise ValueError(
                            f"LLM output is not set for sentence pair {nli_instance.sentence_pair_id}"
                        )

        self._set_is_llm_filtered()
        self._set_llm_output_stat()

    def _set_llm_output_stat(self) -> None:
        if self.is_llm_filtered is False:
            raise ValueError("LLM filtering has not been performed yet.")

        for nli_instance in self.nli_instances:
            # premise
            for aug in nli_instance.sentence1.neg_augmentation.augmented_negs:
                if aug.llm_output == LLM_OUTPUT_CORRECT:
                    aug.set_is_correct(True)
                elif aug.llm_output == LLM_OUTPUT_INCORRECT:
                    aug.set_is_correct(False)
                else:
                    raise ValueError(f"Invalid LLM output: {aug.llm_output}")

                assert aug.neg_position is not None
                self.aug_stat.update_filtering_results(
                    "premise", aug.neg_position, aug.target_pos_simple, aug.llm_output
                )
            # hypothesis
            for aug in nli_instance.sentence2.neg_augmentation.augmented_negs:
                if aug.llm_output == LLM_OUTPUT_CORRECT:
                    aug.set_is_correct(True)
                elif aug.llm_output == LLM_OUTPUT_INCORRECT:
                    aug.set_is_correct(False)
                else:
                    raise ValueError(f"Invalid LLM output: {aug.llm_output}")

                assert aug.neg_position is not None
                self.aug_stat.update_filtering_results(
                    "hypothesis",
                    aug.neg_position,
                    aug.target_pos_simple,
                    aug.llm_output,
                )
            nli_instance.sentence1.neg_augmentation.set_is_correct_num()
            nli_instance.sentence2.neg_augmentation.set_is_correct_num()

    def _check_sentence_naturalness_by_openai_api(
        self, log_output_file_path_name: str, start_idx: int, end_idx: int
    ) -> None:
        if end_idx == -1:
            end_idx = len(self)
        else:
            end_idx += 1

        # initialize OpenAI client
        # (API key should be set in the environment variable)
        openai_api_key = os.getenv("OPEN_AI_API_KEY")
        if openai_api_key is None:
            raise ValueError("OPEN_AI_API_KEY is not set.")

        client = OpenAI(api_key=openai_api_key)

        output_logs: list[dict] = []

        print("Checking sentence naturalness by LLM...")
        print("Start index: ", start_idx, "End index (+-0 or +1): ", end_idx)
        for nli_instance in tqdm(self.nli_instances[start_idx:end_idx]):
            output_dict: dict[str, int | dict[str, str | list]] = {
                "sentence_pair_id": nli_instance.sentence_pair_id,
                "sentence1": {
                    "original": nli_instance.sentence1_original,
                    "augmented": [],
                },
                "sentence2": {
                    "original": nli_instance.sentence2_original,
                    "augmented": [],
                },
            }
            for aug in nli_instance.sentence1.neg_augmentation.augmented_negs:
                try:
                    completion = self._get_llm_output(
                        openai_client=client,
                        sent=aug.augmented_sent(nli_instance.sentence1.nodes),
                        original_sent=nli_instance.sentence1.plain_sentence,
                    )
                    aug.set_llm_output(completion.choices[0].message.content)
                    output_dict["sentence1"]["augmented"].append(  # type: ignore
                        {
                            "id_in_sent": aug.id_in_sent,
                            "augmented_sent": aug.augmented_sent(
                                nli_instance.sentence1.nodes
                            ),
                            "llm_output": completion.choices[0].message.content,
                            "target_pos": aug.target_pos,
                            "neg_position": aug.neg_position,
                        }
                    )  # type: ignore
                except Exception as e:
                    print(e)
                    continue

            for aug in nli_instance.sentence2.neg_augmentation.augmented_negs:
                try:
                    completion = self._get_llm_output(
                        openai_client=client,
                        sent=aug.augmented_sent(nli_instance.sentence2.nodes),
                        original_sent=nli_instance.sentence2.plain_sentence,
                    )
                    aug.set_llm_output(completion.choices[0].message.content)
                    output_dict["sentence2"]["augmented"].append(  # type: ignore
                        {
                            "id_in_sent": aug.id_in_sent,
                            "augmented_sent": aug.augmented_sent(
                                nli_instance.sentence2.nodes
                            ),
                            "llm_output": completion.choices[0].message.content,
                            "target_pos": aug.target_pos,
                            "neg_position": aug.neg_position,
                        }
                    )  # type: ignore
                except Exception as e:
                    print(e)
                    continue

            output_logs.append(output_dict)
            dirname = log_output_file_path_name.split("/")[-1]
            os.makedirs(
                f"logs/llm_filtering_output/llm_all/{dirname}",
                exist_ok=True,
            )
            utils.save_json(
                f"logs/llm_filtering_output/llm_all/{dirname}/sentence_pair_{nli_instance.sentence_pair_id}.json",
                [output_dict],
            )

        # save log files
        utils.save_jsonl(f"{log_output_file_path_name}.jsonl", output_logs)
        print("Saved log jsonl file: ", f"{log_output_file_path_name}.jsonl\n")
        response_log = [
            {"start_idx": start_idx, "end_idx": end_idx, "response": str(completion)}  # type: ignore
        ]
        utils.save_json(f"{log_output_file_path_name}_response.json", response_log)
        print(
            "Saved response log file: ", f"{log_output_file_path_name}_response.json\n"
        )

    def _get_llm_output(self, openai_client: OpenAI, sent: str, original_sent: str):
        messages = [p for p in cfg.llm_api_params["prompt"]]  # type: ignore
        # 1-shot
        messages.append({"role": "user", "content": original_sent})
        messages.append({"role": "assistant", "content": "正しい"})
        messages.append({"role": "user", "content": sent})
        completion = openai_client.chat.completions.create(
            model=cfg.llm_api_params["model_name"],  # type: ignore
            messages=messages,  # type: ignore
            max_tokens=cfg.llm_api_params["max_tokens"],  # type: ignore
            temperature=cfg.llm_api_params["temperature"],  # type: ignore
            seed=cfg.llm_api_params["seed"],  # type: ignore
        )

        return completion

    def make_new_nli_pairs(
        self,
        output_jsonl_path: str,
        sample_num: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """
        Step 2.1
        create new Premise-Hypothesis sentence pairs
            - output_jsonl_path: output file path
            - sample_num: sampling num
            - random_seed: seed for sampling
        """

        idx_all: list[int] = [nli.sentence_pair_id for nli in self.nli_instances]

        if sample_num is None:
            self._make_nli_pairs(idx_all, output_jsonl_path)

        else:
            # load indices of JNLI instances that contain negation cues
            idx_neg: list[int] = self.get_jnli_neg_instance_indices(
                paths.cue_detection_output_files[self.split]
            )
            # indices of JNLI instances that meet the requirements:
            ## neg(p) = 0, neg(h) = 0, |P'| > 0, and |H'| > 0
            idx_not_neg: list[int] = [
                i
                for i in idx_all
                if i not in idx_neg
                and self.get_item_by_sentence_pair_id(
                    i
                ).sentence1.neg_augmentation.is_correct_num  # type: ignore
                > 0
                and self.get_item_by_sentence_pair_id(
                    i
                ).sentence2.neg_augmentation.is_correct_num  # type: ignore
                > 0
            ]

            # shuffle
            idx_not_neg_shuffled: list[int] = [i for i in idx_not_neg]
            random.seed(random_seed)
            random.shuffle(idx_not_neg_shuffled)

            pair_counter: int = 0
            end_idx: int | None = 0
            for i, nli_idx in enumerate(idx_not_neg_shuffled):
                nli_instance = self.get_item_by_sentence_pair_id(nli_idx)
                assert (
                    nli_instance.sentence1.neg_augmentation.is_correct_num
                    and nli_instance.sentence2.neg_augmentation.is_correct_num
                )
                p = nli_instance.sentence1.neg_augmentation.is_correct_num
                h = nli_instance.sentence2.neg_augmentation.is_correct_num
                pair_counter += p + h + p * h
                if pair_counter > sample_num:
                    end_idx = i
                    break

            if end_idx is None:
                raise ValueError("Sample number is too large.")

            idx_not_neg_sampling = [i for i in idx_not_neg_shuffled[: end_idx + 1]]
            idx_not_neg_sampling.sort()

            self._make_nli_pairs(idx_not_neg_sampling, output_jsonl_path)

    def _make_nli_pairs(self, indices: list[int], output_jsonl_path: str) -> None:
        """
        create new (P, H) sentence pairs for JNLI instances whose indices are given
        """
        print("\nMaking new sentence pairs for annotation...")

        output_dicts: list[dict] = []
        counter: int = 0

        for nli_idx in tqdm(indices):
            nli_instance = self.get_item_by_sentence_pair_id(nli_idx)
            assert (
                nli_instance.sentence1.neg_augmentation.is_correct_num is not None
                and nli_instance.sentence2.neg_augmentation.is_correct_num is not None
            )
            if (
                nli_instance.sentence1.neg_augmentation.is_correct_num > 0
                and nli_instance.sentence2.neg_augmentation.is_correct_num > 0
            ):
                new_dict = nli_instance.make_new_sentence_pairs(start_id=counter)
                output_dicts.extend(new_dict)
                counter += len(new_dict)

        utils.save_jsonl(file_path=output_jsonl_path, data=output_dicts)
        print(f"\nSaved new sentence pairs for annotation: {output_jsonl_path}")

    def get_jnli_neg_instance_indices(self, log_file: str | Path) -> list[int]:
        loaded_json = utils.load_jsonl(log_file)
        negation_contained_pairs_idx: list[int] = [
            int(d["sentence_pair_id"]) for d in loaded_json
        ]

        return negation_contained_pairs_idx


class NLIInstance:
    def __init__(
        self,
        sentence_pair_id: str,
        yjcaptions_id: str,
        sentence1: str,
        sentence2: str,
        label: L.NLI_LABEL,
    ):
        self.sentence_pair_id: int = int(sentence_pair_id)
        self.yjcaptions_id: str = yjcaptions_id

        yj_ids = yjcaptions_id.split("-")
        assert len(yj_ids) == 3
        self.image_id: str = yj_ids[0]
        self.sentence1_id: str = yj_ids[1]
        self.sentence2_id: str = yj_ids[2]

        self.sentence1_original = sentence1
        self.sentence2_original = sentence2

        self.sentence1 = Sentence(
            plain_sentence=sentence1,
            sentence_pair_id=self.sentence_pair_id,
            image_id=self.image_id,
            sentence_id=self.sentence1_id,
            hypo_or_prem="premise",
        )

        self.sentence2 = Sentence(
            plain_sentence=sentence2,
            sentence_pair_id=self.sentence_pair_id,
            image_id=self.image_id,
            sentence_id=self.sentence2_id,
            hypo_or_prem="hypothesis",
        )

        self.label = label

    def perform_augmentation_for_jnli_instance(self) -> None:
        self.sentence1.perform_augmentation()
        self.sentence2.perform_augmentation()

    def make_new_sentence_pairs(
        self,
        start_id: int,
    ) -> list[dict[str, int | str | dict[str, str | dict[str, str | int] | None]]]:

        new_pairs: list[
            dict[str, int | str | dict[str, str | dict[str, str | int] | None]]
        ] = []

        if (
            self.sentence1.neg_augmentation.is_correct_num == 0
            or self.sentence2.neg_augmentation.is_correct_num == 0
        ):
            raise ValueError(
                "At least one of the sentences does not have any correct negation."
            )
        else:
            id_count: int = 0

            # (Pneg, H)
            for aug1 in self.sentence1.neg_augmentation.augmented_negs:
                if aug1.is_correct:
                    assert aug1.neg_position is not None
                    new_pairs.append(
                        {
                            "id": start_id + id_count,
                            "original_sentence_pair_id": self.sentence_pair_id,
                            "new_pair_id_in_sentence": id_count,
                            "type": AugSentenceType.P_NEG,
                            "sentence1": {
                                "sentence": aug1.augmented_sent(self.sentence1.nodes),
                                "aug": {
                                    "aug_id": aug1.id_in_sent,
                                    "neg_position": aug1.neg_position,
                                    "target_pos": aug1.target_pos_simple,
                                },
                            },
                            "sentence2": {
                                "sentence": self.sentence2.plain_sentence,
                                "aug": None,
                            },
                        }
                    )
                    id_count += 1

            # (P, Hneg)
            for aug2 in self.sentence2.neg_augmentation.augmented_negs:
                if aug2.is_correct:
                    assert aug2.neg_position is not None
                    new_pairs.append(
                        {
                            "id": start_id + id_count,
                            "original_sentence_pair_id": self.sentence_pair_id,
                            "new_pair_id_in_sentence": id_count,
                            "type": AugSentenceType.H_NEG,
                            "sentence1": {
                                "sentence": self.sentence1.plain_sentence,
                                "aug": None,
                            },
                            "sentence2": {
                                "sentence": aug2.augmented_sent(self.sentence2.nodes),
                                "aug": {
                                    "aug_id": aug2.id_in_sent,
                                    "neg_position": aug2.neg_position,
                                    "target_pos": aug2.target_pos_simple,
                                },
                            },
                        }
                    )
                    id_count += 1

            # (Pneg, Hneg)
            for aug1 in self.sentence1.neg_augmentation.augmented_negs:
                if aug1.is_correct:
                    for aug2 in self.sentence2.neg_augmentation.augmented_negs:
                        if aug2.is_correct:
                            assert (
                                aug1.neg_position is not None
                                and aug2.neg_position is not None
                            )
                            new_pairs.append(
                                {
                                    "id": start_id + id_count,
                                    "original_sentence_pair_id": self.sentence_pair_id,
                                    "new_pair_id_in_sentence": id_count,
                                    "type": AugSentenceType.P_NEG_H_NEG,
                                    "sentence1": {
                                        "sentence": aug1.augmented_sent(
                                            self.sentence1.nodes
                                        ),
                                        "aug": {
                                            "aug_id": aug1.id_in_sent,
                                            "neg_position": aug1.neg_position,  # type: ignore
                                            "target_pos": aug1.target_pos_simple,
                                        },
                                    },
                                    "sentence2": {
                                        "sentence": aug2.augmented_sent(
                                            self.sentence2.nodes
                                        ),
                                        "aug": {
                                            "aug_id": aug2.id_in_sent,
                                            "neg_position": aug2.neg_position,
                                            "target_pos": aug2.target_pos_simple,
                                        },
                                    },
                                }
                            )
                            id_count += 1

        return new_pairs


class Sentence:
    """
    sentence for JNLI
    """

    def __init__(
        self,
        plain_sentence: str,
        sentence_pair_id: int,
        image_id: str,
        sentence_id: str,
        hypo_or_prem: Literal["hypothesis", "premise"],
        clean_text: bool = True,
    ) -> None:
        self.plain_sentence = plain_sentence
        self.sentence_pair_id = sentence_pair_id
        self.image_id = image_id
        self.sentence_id = sentence_id
        self.hypo_or_prem = hypo_or_prem

        # pre-processing
        if clean_text:
            self.clean_text()

        # perform morphological analysis and get a root node
        self.nodes = MorphAnalysis.perform_analysis(self.plain_sentence)

        self.neg_augmentation = NegationAugmentation()

        self.is_augmented: bool | None = None

    def __len__(self) -> int:
        assert len(self.nodes) == self.nodes[-1].position + 1
        return len(self.nodes)

    def is_hand_generated(self) -> bool:
        if "g" in self.sentence_id:
            return True
        else:
            return False

    def clean_text(self) -> None:
        """
        preprocessing for the sentence: correct format inconsistency in JNLI data
        """
        self.plain_sentence = self.plain_sentence.replace("　", " ")
        self.plain_sentence = self.plain_sentence.replace("｡", "。")
        self.plain_sentence = self.plain_sentence.replace("。。", "。")

    def set_is_augmented(self, value: bool) -> None:
        assert self.is_augmented is None
        self.is_augmented = value

    def perform_augmentation(self) -> None:
        """
        Step 1.1
        """
        if self.is_augmented is not None:
            raise ValueError("Augmentation has already been performed.")
        else:
            # perform negation augmentation
            self.neg_augmentation.negative_unidic(self.nodes)
            # update augmentation count
            self.neg_augmentation.set_augmented_num()
            # update augmentation info
            if len(self.neg_augmentation.augmented_negs) == 0:
                self.set_is_augmented(False)
            else:
                self.set_is_augmented(True)

    def display_augmentation_info(self) -> None:
        if self.is_augmented is None:
            raise ValueError("Augmentation has not been performed yet.")
        else:
            print(f"{self.sentence_pair_id=}, {self.hypo_or_prem}")
            print(f"Original: {self.plain_sentence}")

            if self.is_augmented:
                print("Augmented:")
                for aug in self.neg_augmentation.augmented_negs:
                    print(
                        f"{aug.id_in_sent}: {aug.position} ({aug.neg_position}), {aug.target_pos}, {aug.rule}"
                    )
                    print(f"{aug.augmented_sent(self.nodes)}")
                    print()
                print()
            else:
                print("Not augmented.")
                print()


class AugmentationStatistics:
    """
    statistics for Step 1.1 and 1.2
    """

    def __init__(self, original_nli_pairs: int) -> None:
        self.original_nli_pairs: int = original_nli_pairs

        # statistics for Step 1.1
        self.augmentation_results: dict[
            L.NLI_P_OR_H, dict[L.AUGMENTED_NEG_POSITION, dict[L.AUG_TARGET_POS, int]]
        ] = {
            "premise": {
                "clausal": {"動詞": 0, "形容詞": 0, "形容動詞": 0},
                "sub-clausal": {"動詞": 0, "形容詞": 0, "形容動詞": 0},
            },
            "hypothesis": {
                "clausal": {"動詞": 0, "形容詞": 0, "形容動詞": 0},
                "sub-clausal": {"動詞": 0, "形容詞": 0, "形容動詞": 0},
            },
        }

        # statistics for Step 1.2
        self.filtering_results: dict[
            L.NLI_P_OR_H,
            dict[
                L.AUGMENTED_NEG_POSITION,
                dict[L.AUG_TARGET_POS, dict[L.LLM_OUTPUT, int]],
            ],
        ] = {
            "premise": {
                "clausal": {
                    "動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                },
                "sub-clausal": {
                    "動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                },
            },
            "hypothesis": {
                "clausal": {
                    "動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                },
                "sub-clausal": {
                    "動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                    "形容動詞": {LLM_OUTPUT_CORRECT: 0, LLM_OUTPUT_INCORRECT: 0},
                },
            },
        }

    def display_augmentation_details(self) -> None:
        for key_p_or_h in get_args(L.NLI_P_OR_H):
            for key_position in get_args(L.AUGMENTED_NEG_POSITION):
                for key_target_pos in get_args(L.AUG_TARGET_POS):
                    print(
                        f"{key_p_or_h}, {key_position}, {key_target_pos}: {self.augmentation_results[key_p_or_h][key_position][key_target_pos]}"
                    )

    def display_filtering_details(self) -> None:
        for key_p_or_h in get_args(L.NLI_P_OR_H):
            for key_position in get_args(L.AUGMENTED_NEG_POSITION):
                for key_target_pos in get_args(L.AUG_TARGET_POS):
                    print(
                        f"{key_p_or_h}, {key_position}, {key_target_pos}: {self.filtering_results[key_p_or_h][key_position][key_target_pos]}"
                    )

    def update_augmentation_results(
        self,
        p_or_h: L.NLI_P_OR_H,
        position: L.AUGMENTED_NEG_POSITION,
        target_pos: L.AUG_TARGET_POS,
        num: int = 1,
    ) -> None:
        if p_or_h not in get_args(L.NLI_P_OR_H):
            raise ValueError(f"Invalid NLI sentence: {p_or_h}")
        if position not in get_args(L.AUGMENTED_NEG_POSITION):
            raise ValueError(f"Invalid position: {position}")
        if target_pos not in get_args(L.AUG_TARGET_POS):
            raise ValueError(f"Invalid target position: {target_pos}")

        self.augmentation_results[p_or_h][position][target_pos] += num

    def augmented_num_by_p_or_h(self, p_or_h: L.NLI_P_OR_H) -> int:
        if p_or_h not in get_args(L.NLI_P_OR_H):
            raise ValueError(f"Invalid NLI sentence: {p_or_h}")

        return sum(
            [
                self.augmentation_results[p_or_h][position][target_pos]
                for position in get_args(L.AUGMENTED_NEG_POSITION)
                for target_pos in get_args(L.AUG_TARGET_POS)
            ]
        )

    def augmented_num_by_neg_position(self, position: L.AUGMENTED_NEG_POSITION) -> int:
        if position not in get_args(L.AUGMENTED_NEG_POSITION):
            raise ValueError(f"Invalid position: {position}")

        return sum(
            [
                self.augmentation_results[p_or_h][position][target_pos]
                for p_or_h in get_args(L.NLI_P_OR_H)
                for target_pos in get_args(L.AUG_TARGET_POS)
            ]
        )

    def augmented_num_by_target_pos(self, target_pos: L.AUG_TARGET_POS) -> int:
        if target_pos not in get_args(L.AUG_TARGET_POS):
            raise ValueError(f"Invalid target position: {target_pos}")

        return sum(
            [
                self.augmentation_results[p_or_h][position][target_pos]
                for p_or_h in get_args(L.NLI_P_OR_H)
                for position in get_args(L.AUGMENTED_NEG_POSITION)
            ]
        )

    def augmented_num_all(self) -> int:
        if not self._augmented_sum_sanitary_check():
            raise ValueError("Augmented numbers are not consistent.")

        return self.augmented_num_by_p_or_h("premise") + self.augmented_num_by_p_or_h(
            "hypothesis"
        )

    def _augmented_sum_sanitary_check(self) -> bool:
        if (
            self.augmented_num_by_p_or_h("premise")
            + self.augmented_num_by_p_or_h("hypothesis")
            == self.augmented_num_by_neg_position("clausal")
            + self.augmented_num_by_neg_position("sub-clausal")
            == self.augmented_num_by_target_pos("動詞")
            + self.augmented_num_by_target_pos("形容詞")
            + self.augmented_num_by_target_pos("形容動詞")
        ):
            return True
        else:
            return False

    def augmented_num_per_instance(self, sentence_or_nli_pair: "str") -> float:
        if sentence_or_nli_pair == "sentence":
            return self.augmented_num_all() / (self.original_nli_pairs * 2)
        elif sentence_or_nli_pair == "nli_pair":
            return self.augmented_num_all() / self.original_nli_pairs
        else:
            raise ValueError(
                f"Invalid argument: {sentence_or_nli_pair}. Specify 'sentence' or 'nli_pair'."
            )

    def update_filtering_results(
        self,
        p_or_h: L.NLI_P_OR_H,
        position: L.AUGMENTED_NEG_POSITION,
        target_pos: L.AUG_TARGET_POS,
        output: L.LLM_OUTPUT,
        num: int = 1,
    ) -> None:
        if p_or_h not in get_args(L.NLI_P_OR_H):
            raise ValueError(f"Invalid NLI sentence: {p_or_h}")
        if position not in get_args(L.AUGMENTED_NEG_POSITION):
            raise ValueError(f"Invalid position: {position}")
        if target_pos not in get_args(L.AUG_TARGET_POS):
            raise ValueError(f"Invalid target position: {target_pos}")
        if output not in get_args(L.LLM_OUTPUT):
            raise ValueError(f"Invalid output: {output}")

        self.filtering_results[p_or_h][position][target_pos][output] += num

    def filtered_num(self, correct_or_incorrect: str) -> int:
        correct_num = sum(
            [
                self.filtering_results[p_or_h][position][target_pos][LLM_OUTPUT_CORRECT]
                for p_or_h in get_args(L.NLI_P_OR_H)
                for position in get_args(L.AUGMENTED_NEG_POSITION)
                for target_pos in get_args(L.AUG_TARGET_POS)
            ]
        )
        incorrect_num = sum(
            [
                self.filtering_results[p_or_h][position][target_pos][
                    LLM_OUTPUT_INCORRECT
                ]
                for p_or_h in get_args(L.NLI_P_OR_H)
                for position in get_args(L.AUGMENTED_NEG_POSITION)
                for target_pos in get_args(L.AUG_TARGET_POS)
            ]
        )
        # sanity check
        if correct_num + incorrect_num != self.augmented_num_all():
            raise ValueError("Filtered numbers are not consistent.")

        if correct_or_incorrect == "correct":
            return correct_num
        elif correct_or_incorrect == "incorrect":
            return incorrect_num
        else:
            raise ValueError(
                f"Invalid argument: {correct_or_incorrect}. Specify 'correct' or 'incorrect'."
            )


class Args(Tap):
    split: L.SPLIT
    start_idx: int | None = None
    end_idx: int | None = None
    request_api: bool = False


def main(args: Args) -> None:
    jnli_file = paths.jnli_dataset[args.split]
    jnli = JNLIDataset(jnli_file, split=args.split)
    # Step 1.1
    jnli.perform_augmentation()

    # Step 1.2
    datetime = utils.get_current_datetime().strftime("%y%m%d%H%M%S")
    concat_log_file = (
        f"logs/llm_filtering_output/llm_all/{datetime}_jnli_{args.split}_start{args.start_idx}_end{args.end_idx}"
        if args.request_api
        else paths.llm_filtering_output_files[args.split]
    )
    jnli.check_sentence_naturalness(
        request_api=args.request_api,
        log_output_file_path_name=concat_log_file,
        start_idx=args.start_idx if args.start_idx else 0,
        end_idx=args.end_idx if args.end_idx else -1,
    )

    # Step 2.1
    jnli.make_new_nli_pairs(
        output_jsonl_path=paths.annotation_target_files[args.split],
        sample_num=cfg.annotation_sampling_nums[args.split],
        random_seed=cfg.annotation_sampling_random_seed,
    )


if __name__ == "__main__":
    main(args=Args().parse_args())
