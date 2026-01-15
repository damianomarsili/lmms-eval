import re
from typing import Any, Dict, Optional


def countbenchqa_doc_to_visual(doc: Dict[str, Any]) -> list:
    image = doc["image"]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return [image]


def countbenchqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    question = doc["question"].strip()

    parts = [pre_prompt, question, post_prompt]
    return "\n".join([p for p in parts if p])


def countbenchqa_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["number"])


def _extract_last_int(text: str) -> Optional[int]:
    text = _strip_think_prefix(text)
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def countbenchqa_process_results(doc: Dict[str, Any], results: list[str]) -> Dict[str, float]:
    """
    Accept model free-form output and grade by the last integer it contains.
    """
    prediction_raw = _strip_think_prefix(results[0])
    pred_int = _extract_last_int(prediction_raw)
    gold_int = doc["number"]
    return {"exact_match": 1.0 if pred_int is not None and pred_int == gold_int else 0.0}


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <plan>...</plan> or <think>...</think> block if present and return the remainder.
    If only a closing tag is present, take text after it.
    """
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)
