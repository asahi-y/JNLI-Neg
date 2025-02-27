from typing import Literal
from morph_analysis import MorphNodes
import literals as L


class NegationAugmentation:
    """
    negation augmentation for a sentence
    """

    def __init__(self):
        self.augmented_negs: list[AugmentedNeg] = []

        # num of negation augmentation in a sentence
        self.augmented_num: int | None = None

        # num of correct (judged by LLM) negation augmentation in a sentence
        self.is_correct_num: int | None = None

    def set_augmented_num(self) -> None:
        self.augmented_num = len(self.augmented_negs)

    def set_is_correct_num(self) -> None:
        self.is_correct_num = sum([1 for aug in self.augmented_negs if aug.is_correct])

    def _set_neg_positions(self, nodes: MorphNodes) -> None:
        assert nodes[-1].word.surface == "。"
        end_word_pos = nodes[-1].position - 1
        for aug in self.augmented_negs:
            if aug.position == end_word_pos:
                aug.set_neg_position("clausal")
            else:
                aug.set_neg_position("sub-clausal")

    def negative_unidic(self, nodes: MorphNodes) -> None:
        """
        negation augmentation rules based on UniDic
        """
        a_mapping = {
            "う": "わ",
            "つ": "た",
            "る": "ら",
            "む": "ま",
            "ぶ": "ば",
            "ぬ": "な",
            "ぐ": "が",
            "す": "さ",
            "く": "か",
        }

        result = ""
        rendaku_flag = False
        vanish_flag = False
        augment_id_in_sent: int = 0
        node = nodes.head
        while node:
            orthBase = node.word.orthBase if node.word.orthBase else node.word.surface
            # JAPANESE: 動詞に対する処理
            if node.word.pos1 == "動詞":
                if node.word.cForm[0:3] in ["終止形", "連体形"]:
                    if node.word.cType[0:3] in ["上一段", "下一段"]:
                        result += orthBase[:-1] + "ない"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"{node.word.cForm[0:3]}-{node.word.cType[0:3]}",
                                orthBase[:-1] + "ない",
                            )
                        )
                        augment_id_in_sent += 1
                    elif orthBase[-2:] == "ある":
                        result += orthBase[:-2] + "ない"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"{node.word.cForm[0:3]}-ある",
                                orthBase[:-2] + "ない",
                            )
                        )
                        augment_id_in_sent += 1
                    elif orthBase[-2:] == "有る":
                        result += orthBase[:-2] + "無い"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"{node.word.cForm[0:3]}-有る",
                                orthBase[:-2] + "無い",
                            )
                        )
                        augment_id_in_sent += 1
                    elif node.word.cType[0:2] == "五段":
                        if orthBase[-1] in a_mapping.keys():
                            result += orthBase[:-1] + a_mapping[orthBase[-1]] + "ない"
                            self.augmented_negs.append(
                                AugmentedNeg(
                                    augment_id_in_sent,
                                    node.position,
                                    "動詞",
                                    f"五段",
                                    orthBase[:-1] + a_mapping[orthBase[-1]] + "ない",
                                )
                            )
                            augment_id_in_sent += 1
                        else:
                            result += node.word.surface
                    elif node.word.cType == "カ行変格":
                        result += "来ない"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                "カ行変格",
                                "来ない",
                            )
                        )
                        augment_id_in_sent += 1
                    elif "サ行変格" in node.word.cType:
                        result += "しない"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                "サ行変格",
                                "しない",
                            )
                        )
                        augment_id_in_sent += 1
                    else:
                        raise ValueError()

                # JAPANESE:「連用形＋た」の否定形を作る
                elif node.next and node.next.word.surface in ["た", "だ"]:
                    if node.next.word.cForm[0:3] == "終止形":
                        if node.word.cForm == "連用形-イ音便":
                            if node.word.cType == "五段-カ行":
                                result += orthBase[:-1] + "かなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-イ音便-五段-カ行",
                                        orthBase[:-1] + "かなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-ガ行":
                                rendaku_flag = True
                                result += orthBase[:-1] + "がなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-イ音便-五段-ガ行",
                                        orthBase[:-1] + "がなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                        elif node.word.cForm == "連用形-ウ音便":
                            result += orthBase[:-1] + "わなかっ"
                            self.augmented_negs.append(
                                AugmentedNeg(
                                    augment_id_in_sent,
                                    node.position,
                                    "動詞",
                                    "「連用形 + た」-終止形-連用形-ウ音便",
                                    orthBase[:-1] + "わなかっ",
                                )
                            )
                            augment_id_in_sent += 1
                        elif node.word.cForm == "連用形-促音便":
                            if node.word.cType == "五段-ワア行":
                                result += orthBase[:-1] + "わなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-五段-ワア行",
                                        orthBase[:-1] + "わなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-タ行":
                                result += orthBase[:-1] + "たなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-五段-タ行",
                                        orthBase[:-1] + "たなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif orthBase[-2:] == "ある":
                                result += orthBase[:-2] + "なかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-ある",
                                        orthBase[:-2] + "なかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif orthBase[-2:] == "有る":
                                result += orthBase[:-2] + "無かっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-有る",
                                        orthBase[:-2] + "無かっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-ラ行":
                                result += orthBase[:-1] + "らなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-五段-ラ行",
                                        orthBase[:-1] + "らなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-カ行":
                                result += orthBase[:-1] + "かなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-促音便-五段-カ行",
                                        orthBase[:-1] + "かなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                        elif node.word.cForm == "連用形-撥音便":
                            rendaku_flag = True
                            if node.word.cType[0:5] == "五段-マ行":
                                result += orthBase[:-1] + "まなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-撥音便-五段-マ行",
                                        orthBase[:-1] + "まなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-ナ行":
                                result += orthBase[:-1] + "ななかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-撥音便-五段-ナ行",
                                        orthBase[:-1] + "ななかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-バ行":
                                result += orthBase[:-1] + "ばなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        "「連用形 + た」-終止形-連用形-撥音便-五段-バ行",
                                        orthBase[:-1] + "ばなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                        elif node.word.cForm[0:3] == "連用形":
                            if node.word.cType[0:3] in ["上一段", "下一段"]:
                                result += orthBase[:-1] + "なかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        f"「連用形 + た」-終止形-連用形-{node.word.cType[0:3]}",
                                        orthBase[:-1] + "なかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType[0:5] == "五段-サ行":
                                result += orthBase[:-1] + "さなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        f"「連用形 + た」-終止形-連用形-五段-サ行",
                                        orthBase[:-1] + "さなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.cType == "サ行変格":
                                result += orthBase[:-2] + "しなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "動詞",
                                        f"「連用形 + た」-終止形-連用形-サ行変格",
                                        orthBase[:-2] + "しなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            else:
                                result += node.word.surface
                        else:
                            result += node.word.surface
                    elif node.next.word.cForm[0:3] == "連体形":
                        result += node.word.surface
                elif node.next and node.next.word.surface in ["て", "で"]:
                    vanish_flag = True
                    if node.word.cType[0:3] in ["上一段", "下一段"]:
                        result += orthBase[:-1] + "ず"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"{node.word.cType[0:3]}-{node.next.word.surface}",
                                orthBase[:-1] + "ず",
                            )
                        )
                        augment_id_in_sent += 1
                    elif orthBase[-2:] == "ある":
                        result += orthBase[:-2] + "なくて"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"ある-{node.next.word.surface}",
                                orthBase[:-2] + "なくて",
                            )
                        )
                        augment_id_in_sent += 1
                    elif orthBase[-2:] == "有る":
                        result += orthBase[:-2] + "無くて"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"有る-{node.next.word.surface}",
                                orthBase[:-2] + "無くて",
                            )
                        )
                        augment_id_in_sent += 1
                    elif node.word.cType[0:2] == "五段":
                        result += orthBase[:-1] + a_mapping[orthBase[-1]] + "ず"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"五段-{node.next.word.surface}",
                                orthBase[:-1] + a_mapping[orthBase[-1]] + "ず",
                            )
                        )
                        augment_id_in_sent += 1
                    elif node.word.cType == "カ行変格":
                        result += "来ず"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"カ行変格-{node.next.word.surface}",
                                "来ず",
                            )
                        )
                        augment_id_in_sent += 1
                    elif node.word.cType == "サ行変格":
                        result += "せず"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞",
                                f"サ行変格-{node.next.word.surface}",
                                "せず",
                            )
                        )
                        augment_id_in_sent += 1
                else:
                    result += node.word.surface
            elif node.word.pos1 == "助詞":
                if vanish_flag:
                    vanish_flag = False
                    result += ""
                    self.augmented_negs[-1].process_vanish_flag()
                else:
                    result += orthBase
            elif node.word.pos1 == "助動詞":
                # JAPANESE:「飛んだ」->「飛ばなかった」など連濁に対応
                if rendaku_flag:
                    rendaku_flag = False
                    result += "た"
                    self.augmented_negs[-1].add_new_text("た")
                elif node.word.cType == "助動詞-レル" or orthBase in ["させる", "せる"]:
                    if node.word.cForm[0:3] in ["終止形", "連体形"]:
                        result += orthBase[:-1] + "ない"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞-助動詞",
                                f"「れる」「られる」or「せる」「させる」-{node.word.cForm[0:3]}",
                                orthBase[:-1] + "ない",
                            )
                        )
                        augment_id_in_sent += 1
                    elif node.next and node.next.word.surface == "た":
                        result += orthBase[:-1] + "なかっ"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞-助動詞",
                                f"「れる」「られる」or「せる」「させる」+「た」",
                                orthBase[:-1] + "なかっ",
                            )
                        )
                        augment_id_in_sent += 1
                    else:
                        result += node.word.surface
                elif node.word.cType == "助動詞-マス":
                    if node.word.surface == "ます":
                        result += "ません"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞-助動詞",
                                "「ます」",
                                "ません",
                            )
                        )
                        augment_id_in_sent += 1
                    else:
                        result += "ませんでし"
                        self.augmented_negs.append(
                            AugmentedNeg(
                                augment_id_in_sent,
                                node.position,
                                "動詞-助動詞",
                                "「まし」",
                                "ませんでし",
                            )
                        )
                        augment_id_in_sent += 1
                # JAPANESE:「た」の連体形の処理
                elif node.word.cType == "助動詞-タ":
                    if node.prev:
                        if node.prev.word.pos1 == "動詞":
                            if node.word.cForm[0:3] == "連体形":
                                if (
                                    node.prev.word.orthBase
                                    and node.prev.word.orthBase[-2:] == "ある"
                                ):
                                    result = result[:-2] + "なかった"
                                    self.augmented_negs.append(
                                        AugmentedNeg(
                                            augment_id_in_sent,
                                            node.position,
                                            "動詞-助動詞",
                                            "「た」-連体形-直前動詞「ある」",
                                            result[:-2] + "なかった",
                                        )
                                    )
                                    augment_id_in_sent += 1
                                elif (
                                    node.prev.word.orthBase
                                    and node.prev.word.orthBase[-2:] == "有る"
                                ):
                                    result = result[:-2] + "無かった"
                                    self.augmented_negs.append(
                                        AugmentedNeg(
                                            augment_id_in_sent,
                                            node.position,
                                            "動詞-助動詞",
                                            "「た」-連体形-直前動詞「有る」",
                                            result[:-2] + "無かった",
                                        )
                                    )
                                    augment_id_in_sent += 1
                                elif node.word.orth == "た":
                                    result += "ていない"
                                    self.augmented_negs.append(
                                        AugmentedNeg(
                                            augment_id_in_sent,
                                            node.position,
                                            "動詞-助動詞",
                                            "「た」-連体形-直前動詞",
                                            "ていない",
                                        )
                                    )
                                    augment_id_in_sent += 1
                                else:
                                    result += "でいない"
                                    self.augmented_negs.append(
                                        AugmentedNeg(
                                            augment_id_in_sent,
                                            node.position,
                                            "動詞-助動詞",
                                            "「た」以外-連体形-直前動詞",
                                            "でいない",
                                        )
                                    )
                                    augment_id_in_sent += 1
                            else:
                                if node.word.orth:
                                    result += node.word.orth
                                else:
                                    result += node.word.surface
                        else:
                            result += node.word.surface
                    else:
                        result += node.word.surface
                elif node.word.cType == "助動詞-ダ":
                    if node.prev:
                        if node.prev.word.pos1 == "形状詞":
                            if node.word.surface in ["だ", "な"]:
                                result += "でない"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "形状詞-助動詞",
                                        "助動詞-ダ-「だ」「な」-直前形状詞",
                                        "でない",
                                    )
                                )
                                augment_id_in_sent += 1
                            elif node.word.surface == "だっ":
                                result += "でなかっ"
                                self.augmented_negs.append(
                                    AugmentedNeg(
                                        augment_id_in_sent,
                                        node.position,
                                        "形状詞-助動詞",
                                        "助動詞-ダ-「だっ」-直前形状詞",
                                        "でなかっ",
                                    )
                                )
                                augment_id_in_sent += 1
                            else:
                                result += node.word.surface
                        else:
                            result += node.word.surface
                    else:
                        result += node.word.surface
                else:
                    result += node.word.surface
            # JAPANESE: 形容詞に対する処理
            elif node.word.pos1 == "形容詞":
                if node.word.lemma == "無い":
                    result += node.word.surface
                elif node.word.cForm[0:3] in ["終止形", "連体形"]:
                    result += orthBase[:-1] + "くない"
                    self.augmented_negs.append(
                        AugmentedNeg(
                            augment_id_in_sent,
                            node.position,
                            "形容詞",
                            f"{node.word.cForm[0:3]}",
                            orthBase[:-1] + "くない",
                        )
                    )
                    augment_id_in_sent += 1
                elif (
                    node.word.cForm == "連用形-促音便"
                    and node.next
                    and node.next.word.surface == "た"
                ):
                    result += orthBase[:-1] + "くなかっ"
                    self.augmented_negs.append(
                        AugmentedNeg(
                            augment_id_in_sent,
                            node.position,
                            "形容詞",
                            "連用形-促音便-直後「た」",
                            orthBase[:-1] + "くなかっ",
                        )
                    )
                    augment_id_in_sent += 1
                else:
                    result += node.word.surface
            else:
                result += node.word.surface
            node = node.next

        self._set_neg_positions(nodes)


class AugmentedNeg:
    """
    information of one augmentation
    (not for one sentence)
    """

    def __init__(
        self,
        id_in_sent: int,
        position: int,
        target_pos: str,
        rule: str,
        new_text: str,
    ) -> None:
        # augmentation id in a sentence
        self.id_in_sent = id_in_sent
        # position of the augmentation target word
        self.position = position
        # part of speech of the augmentation target word
        self.target_pos = target_pos

        self.target_pos_simple: L.AUG_TARGET_POS
        if target_pos == "動詞" or target_pos == "動詞-助動詞":
            self.target_pos_simple = "動詞"
        elif target_pos == "形容詞":
            self.target_pos_simple = "形容詞"
        elif target_pos == "形状詞-助動詞":
            self.target_pos_simple = "形容動詞"
        else:
            raise ValueError(f"Invalid target_pos {target_pos}")

        # applied rule
        self.rule = rule

        self.new_text = new_text
        self.rendaku_flag: bool = False
        self.vanish_flag: bool = False

        self.neg_position: L.AUGMENTED_NEG_POSITION | None = None

        self.llm_output: str | None = None
        self.is_correct: bool | None = None

    def set_llm_output(self, output: str | None) -> None:
        if self.llm_output is not None:
            raise ValueError("llm_output is already set.")

        if output is None:
            print("Warning: LLM output is None!!")
            print()
        self.llm_output = output

    def set_is_correct(self, is_correct: bool) -> None:
        if self.is_correct is not None:
            raise ValueError("is_correct is already set.")
        self.is_correct = is_correct

    def set_neg_position(self, position: Literal["clausal", "sub-clausal"]) -> None:
        if self.neg_position is not None:
            raise ValueError("neg_position is already set.")
        self.neg_position = position

    def add_new_text(self, new_text: str) -> None:
        self.new_text += new_text
        self.rendaku_flag = True

    def process_vanish_flag(self) -> None:
        self.vanish_flag = True

    def augmented_sent(self, nodes: MorphNodes) -> str:
        """
        augmented sentence (plain text)
        """
        result = ""
        node = nodes.head
        while node:
            if node.position == self.position:
                result += self.new_text
            elif self.rendaku_flag and node.position == self.position + 1:
                result += ""
            elif self.vanish_flag and node.position == self.position + 1:
                result += ""
            else:
                result += node.word.surface
            node = node.next

        return result
