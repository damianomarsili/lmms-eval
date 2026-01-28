import re
from typing import Any, Dict, List, Optional


MRA_THRESHOLDS = [0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
YES_SET = {"yes", "y", "yeah", "yep", "true", "1"}
NO_SET = {"no", "n", "nope", "false", "0"}


def omni3d_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    image = doc["image"]
    if hasattr(image, "convert"):
        image = image.convert("RGB")
    return [image]


def omni3d_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    parts = [pre_prompt, question, post_prompt]
    return "\n".join([p for p in parts if p])


def omni3d_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc["answer"])


def _normalize_yes_no(text: str) -> str:
    text = _strip_think_prefix(text)
    cleaned = text.strip().lower()
    if not cleaned:
        return ""
    match = re.findall(r"\b(yes|no)\b", cleaned)
    if match:
        return match[-1]
    token = re.sub(r"[^a-z0-9]+", "", cleaned)
    if token in YES_SET:
        return "yes"
    if token in NO_SET:
        return "no"
    parts = cleaned.split()
    return parts[0] if parts else ""


def _extract_first_number(text: str) -> Optional[float]:
    text = _strip_think_prefix(text)
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _extract_last_answer_tag(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].strip()


def _mra_score(gt: float, pred: float) -> float:
    if gt == 0:
        rel_err = abs(pred)
    else:
        rel_err = abs(gt - pred) / abs(gt)
    hits = [1.0 if rel_err < t else 0.0 for t in MRA_THRESHOLDS]
    return sum(hits) / len(hits)


def omni3d_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Optional[float]]:
    prediction = _strip_think_prefix(results[0] if results else "")
    tagged_answer = _extract_last_answer_tag(prediction)
    if tagged_answer is not None:
        prediction = tagged_answer

    ans_type = str(doc.get("answer_type", "")).lower()
    gt_raw = str(doc.get("answer", "")).strip()

    yes_no_score = None
    multi_score = None
    num_ct_score = None
    num_other_score = None
    overall_score = 0.0

    if ans_type == "int":
        gt_num = None
        try:
            gt_num = int(gt_raw)
        except Exception:
            pass
        pred_num = _extract_first_number(prediction)
        if gt_num is not None and pred_num is not None:
            try:
                pred_int = int(pred_num)
                num_ct_score = 1.0 if pred_int == gt_num else 0.0
            except Exception:
                num_ct_score = 0.0
        else:
            num_ct_score = 0.0
        overall_score = num_ct_score
    elif ans_type == "str":
        gt_lower = gt_raw.lower()
        pred_lower = prediction.strip().lower()
        if gt_lower in {"yes", "no"}:
            normalized_pred = _normalize_yes_no(prediction)
            yes_no_score = 1.0 if normalized_pred == gt_lower else 0.0
            overall_score = yes_no_score
        else:
            multi_score = 1.0 if pred_lower == gt_lower else 0.0
            overall_score = multi_score
    elif ans_type == "float":
        gt_num = None
        try:
            gt_num = float(gt_raw)
        except Exception:
            pass
        pred_num = _extract_first_number(prediction)
        if gt_num is not None and pred_num is not None:
            num_other_score = _mra_score(gt_num, pred_num)
        else:
            num_other_score = 0.0
        overall_score = num_other_score
    else:
        overall_score = 0.0

    return {
        "omni3d_yes_no": yes_no_score,
        "omni3d_multi": multi_score,
        "omni3d_num_ct": num_ct_score,
        "omni3d_num_other": num_other_score,
        "omni3d_overall": overall_score,
    }


def _mean_ignore_none(values: List[Optional[float]]) -> float:
    valid = [v for v in values if v is not None]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def omni3d_aggregate_yes_no(results: List[Optional[float]]) -> float:
    return _mean_ignore_none(results)


def omni3d_aggregate_multi(results: List[Optional[float]]) -> float:
    return _mean_ignore_none(results)


def omni3d_aggregate_num_ct(results: List[Optional[float]]) -> float:
    return _mean_ignore_none(results)


def omni3d_aggregate_num_other(results: List[Optional[float]]) -> float:
    return _mean_ignore_none(results)


def omni3d_aggregate_overall(results: List[Optional[float]]) -> float:
    return _mean_ignore_none(results)


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <plan>...</plan> or <think>...</think> block if present.
    If only a closing tag is present, take text after it.
    """
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)
