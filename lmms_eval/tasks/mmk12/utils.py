import re

from lmms_eval.tasks._task_utils.math_verify_utils import (
    StringExtractionConfig,
    parse,
    verify,
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_CHOICE_CONFIG = [StringExtractionConfig(strings=("A", "B", "C", "D", "E"), try_extract_without_anchor=True, lowercase=False)]


def _extract_answer_tag_content(text):
    match = _ANSWER_TAG_RE.search(str(text))
    if match is None:
        return None
    return match.group(1).strip()


def _parse_choice(text):
    tagged = _extract_answer_tag_content(text)
    text_to_parse = tagged if tagged is not None else str(text)
    return parse(
        text_to_parse,
        extraction_config=_CHOICE_CONFIG,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


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
    prediction = _parse_choice(results[0])
    target = _parse_choice(doc["answer"])
    if len(prediction) == 0 or len(target) == 0:
        return {"exact_match": 0.0}
    return {"exact_match": float(verify(target, prediction, strict=True))}
