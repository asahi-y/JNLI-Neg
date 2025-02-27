from typing import Literal

NLI_LABEL = Literal["entailment", "neutral", "contradiction"]
SPLIT = Literal["train", "val", "test"]
NLI_P_OR_H = Literal["premise", "hypothesis"]

JNLI_ORIGINAL = Literal["original"]
NLI_AUG_COMB_PATTERN = Literal["p_neg", "h_neg", "p_neg_h_neg"]

# clausal: end; sub-clausal: middle
AUGMENTED_NEG_POSITION = Literal["clausal", "sub-clausal"]

AUGMENTED_NEG_POSITION_JNLI_NEG = Literal["end", "mid"]
AUG_TARGET_POS = Literal["動詞", "形容詞", "形容動詞"]

LLM_OUTPUT = Literal["正しい", "正しくない"]

ANNOTATOR = Literal["annotator01", "annotator02", "annotator03"]

# PH-PnegH: ((P, H), (Pneg, H)) -> element of M_{p}
# PH-PHneg: ((P, H), (P, Hneg)) -> element of M_{h}
# PHneg-PnegHneg: ((P, Hneg), (Pneg, Hneg)) -> element of M_{p,ph}
# PnegH-PnegHneg: ((Pneg, H), (Pneg, Hneg)) -> element of M_{h,ph}
MINIMAL_PAIR_TYPE = Literal["PH-PnegH", "PH-PHneg", "PHneg-PnegHneg", "PnegH-PnegHneg"]
