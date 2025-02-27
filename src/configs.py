import literals as L

instruction = """
次の日本語が文法的・意味的に正しいかどうかを判定してください。
正しい場合は「正しい」、そうでない場合は「正しくない」と出力してください。
「正しい」「正しくない」のいずれかのみを出力することを遵守してください。
"""

base_prompt = [
    {"role": "user", "content": f"{instruction}"},
]

llm_api_params = {
    "model_name": "gpt-4o",
    "prompt": base_prompt,
    "max_tokens": 32,
    "temperature": 0.0,
    "seed": 0,
}

annotation_sampling_nums: dict[L.SPLIT, int] = {
    "train": 4800,
    "val": 1200,
}

annotation_sampling_random_seed: int = 0
