import MeCab


class MorphAnalysis:
    @classmethod
    def perform_analysis(
        cls, sentence: str, unidic_path: str | None = None
    ) -> "MorphNodes":
        """
        NOTES:
        - Use MeCab + UniDic to analyze
        - If the default dictionary of the execution environment is not UniDic,
            specify the path (directory) of UniDic with unidic_path.
            Example of UniDic path:
                - /usr/local/lib/mecab/dic/unidic
                - /opt/homebrew/lib/mecab/dic/unidic
        """
        tagger = (
            MeCab.Tagger(f"-Owakati -d {unidic_path}")
            if unidic_path
            else MeCab.Tagger("-Owakati")
        )
        tagger.parseToNode("")
        node = tagger.parseToNode(sentence)

        morph_nodes = MorphNodes()
        while node:
            if "BOS/EOS" in node.feature:
                node = node.next
                continue
            else:
                word = Word(node.feature, node.surface)
                morph_nodes.append_to_tail(word)
                node = node.next

        return morph_nodes


class MorphNode:
    def __init__(self, word: "Word", position: int) -> None:
        self.word = word
        self.position = position
        self.prev: MorphNode | None = None
        self.next: MorphNode | None = None


class MorphNodes:
    """
    Nodes for morphological analysis results
    """

    def __init__(self) -> None:
        self.head: MorphNode | None = None

    def append_to_tail(self, word: "Word") -> None:
        if self.head is None:
            new_node = MorphNode(word, 0)
            self.head = new_node
        else:
            node = self.head
            while node.next is not None:
                node = node.next
            new_node = MorphNode(word, node.position + 1)
            node.next = new_node
            new_node.prev = node

    def display(self):
        current = self.head
        while current:
            print(current.word.surface, end=" ")
            current = current.next
        print()

    def display_detail(self):
        current = self.head
        print(
            "position: surface\tpos1, pos2, pos3, pos4, cType, cForm, lemma, orth, pron, orthBase"
        )
        while current:
            print(
                f"{current.position}: {current.word.surface}\t{current.word.pos1}, {current.word.pos2}, {current.word.pos3}, {current.word.pos4}, {current.word.cType}, {current.word.cForm}, {current.word.lemma}, {current.word.orth}, {current.word.pron}, {current.word.orthBase}"
            )
            current = current.next
        print(f"len: {len(self)}")
        print()

    def __iter__(self):
        current = self.head
        while current:
            yield current
            current = current.next

    def __len__(self) -> int:
        """
        Get the number of nodes
        """
        count: int = 0
        for _ in self:
            count += 1
        return count

    def __getitem__(self, index: int) -> "MorphNode":

        size = len(self)

        if index >= 0:
            if index >= size:
                raise IndexError(f"index {index} out of range!!")
            current = self.head
            count = 0
            while current:
                if count == index:
                    return current
                current = current.next
                count += 1

        else:
            index = size + index
            if index < 0:
                raise IndexError(f"index {index} out of range!!")
            current = self.head
            count = 0
            while current:
                if count == index:
                    return current
                current = current.next
                count += 1

        raise IndexError(f"index {index} out of range!!")


class Word:
    def __init__(self, feature: str, surface: str) -> None:
        """
        feature: MeCab.Node.feature (str)
        """
        self.surface = surface

        features = feature.split(",")
        # JAPANESE: 品詞大分類
        self.pos1 = features[0]
        # JAPANESE: 品詞中分類
        self.pos2 = features[1]
        self.pos3 = features[2]
        self.pos4 = features[3]
        # JAPANESE: 活用型
        self.cType = features[4]
        # JAPANESE: 活用形
        self.cForm = features[5]

        # JAPANESE: 語彙素読み
        self.lForm = None if len(features) <= 6 else features[6]
        # JAPANESE: 語彙素
        self.lemma = None if len(features) <= 7 else features[7]
        # JAPANESE: 正書形出現形
        self.orth = None if len(features) <= 8 else features[8]
        # JAPANESE: 発音形出現形
        self.pron = None if len(features) <= 9 else features[9]
        # JAPANESE: 正書形基本形
        self.orthBase = None if len(features) <= 10 else features[10]
