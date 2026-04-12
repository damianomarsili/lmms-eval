import re

_TAGGED_CHOICE_RE = re.compile(r"<answer>\s*([A-E])\s*</answer>", flags=re.IGNORECASE)
_PREFIXED_CHOICE_RE = re.compile(r"\b(?:answer|option)\s*(?:is\s*)?[:：]?\s*([A-E])\b", flags=re.IGNORECASE)
_PAREN_CHOICE_RE = re.compile(r"\(([A-E])\)", flags=re.IGNORECASE)
_BARE_CHOICE_RE = re.compile(r"^\s*([A-E])\s*[\.\)]?\s*$", flags=re.IGNORECASE)


def _extract_choice(text):
    text = str(text).strip()
    for pattern in (_TAGGED_CHOICE_RE, _PREFIXED_CHOICE_RE, _PAREN_CHOICE_RE, _BARE_CHOICE_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def mmk12_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmk12_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def mmk12_process_results(doc, results):
    prediction = _extract_choice(results[0])
    target = str(doc["answer"]).strip().upper()
    return {"exact_match": float(prediction == target)}
