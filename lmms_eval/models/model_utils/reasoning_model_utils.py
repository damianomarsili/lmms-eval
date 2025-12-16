import re


def parse_reasoning_model_answer(model_answer: str) -> str:
    boxed_answer_match = re.search(r"\\boxed\{\s*(.*?)\s*\}", model_answer, re.DOTALL)
    if boxed_answer_match:
        return boxed_answer_match.group(1)
    else:
        return model_answer
