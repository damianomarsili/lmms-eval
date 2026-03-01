import datetime
import json
import os
import re
from collections import defaultdict
from typing import Dict

from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

eval_type_dict = {
    "coarse perception": ["image scene and topic", "image style & quality", "image emotion"],
    "fine-grained perception": ["object counting", "recognition", "localization"],
    "instance reasoning": ["single-instance reasoning", "cross-instance attribute reasoning", "cross-instance relation reasoning"],
    "logical reasoning": ["code & sequence reasoning", "diagram reasoning", "common reasoning"],
    "science & technology": ["biology & chemistry & physics", "electronics & energy & mechanical eng.", "geography & earth science & agriculture"],
    "math": ["geometry", "numeric commonsense and calculation", "statistical reasoning"],
}


replace_prompt = " Please answer yes or no."


def mmstar_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def mmstar_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = question.replace(replace_prompt, "")
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def exact_match(pred, gt):
    """Brought from MMStar"""
    answer = gt.lower().replace("\n", " ").strip()
    predict_raw = _clean_prediction_text(pred).lower().replace("\n", " ").strip()

    if len(answer) == 1 and answer.isalpha():
        pred_letter = _extract_predicted_letter(predict_raw, {answer})
        return 1.0 if pred_letter == answer else 0.0

    if answer == predict_raw:
        return 1.0
    return 0.0


def _clean_prediction_text(text: str) -> str:
    text = _strip_think_prefix(text)
    # Keep the content and remove XML-like wrappers such as <answer>...</answer>
    text = re.sub(r"</?[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_choice_label(text: str, valid_letters):
    cleaned = text.strip()
    if not cleaned:
        return cleaned
    letters = "".join(sorted({str(letter).lower() for letter in valid_letters if str(letter)}))
    if not letters:
        return cleaned

    patterns = [
        rf"^\(\s*[{letters}]\s*\)\s*",
        rf"^\[\s*[{letters}]\s*\]\s*",
        rf"^[{letters}]\s*[\.\):\-]\s*",
        rf"^[{letters}]\s+",
        r"^option\s+[a-z]\s*[\.\):\-]?\s*",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if updated != cleaned:
            return updated.strip()
    return cleaned


def _extract_option_map_from_question(question: str) -> Dict[str, str]:
    option_map: Dict[str, str] = {}
    if not isinstance(question, str) or not question.strip():
        return option_map

    # Inline format: "(A) ... (B) ... (C) ..."
    inline_matches = re.findall(r"\(([A-Z])\)\s*(.*?)(?=(?:\s*\([A-Z]\)\s*)|$)", question, flags=re.IGNORECASE | re.DOTALL)
    for letter, text in inline_matches:
        normalized = _normalize_text(text)
        if normalized and letter.lower() not in option_map:
            option_map[letter.lower()] = normalized

    if len(option_map) >= 2:
        return option_map

    # Inline format: "Options: A: ..., B: ..., C: ..."
    options_inline = re.search(r"options?\s*:\s*(.+)$", question, flags=re.IGNORECASE | re.DOTALL)
    options_text = options_inline.group(1) if options_inline else question
    # Trim common trailing instruction lines appended after options.
    options_text = re.split(r"\n\s*(?:answer\s+with|please\s+answer)\b", options_text, flags=re.IGNORECASE)[0]
    options_text = options_text.strip()
    option_map = {}
    colon_inline_matches = re.findall(
        r"(?:^|[\n,;]\s*)([A-Z])\s*[:\.\)]\s*(.*?)(?=(?:[\n,;]\s*[A-Z]\s*[:\.\)]\s*)|$)",
        options_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for letter, text in colon_inline_matches:
        normalized = _normalize_text(text)
        if normalized:
            option_map[letter.lower()] = normalized
    if len(option_map) >= 2:
        return option_map

    # Line format: "A. ...", "B) ...", "C: ..."
    option_map = {}
    for line in question.splitlines():
        match = re.match(r"^\s*([A-Z])[\.\):\-]\s*(.+)\s*$", line, flags=re.IGNORECASE)
        if match:
            letter, text = match.group(1).lower(), match.group(2)
            normalized = _normalize_text(text)
            if normalized:
                option_map[letter] = normalized
    return option_map if len(option_map) >= 2 else {}


def _extract_predicted_letter(prediction: str, valid_letters) -> str:
    predict_raw = _clean_prediction_text(prediction).lower().replace("\n", " ").strip()
    if not predict_raw:
        return ""

    valid_letters = {str(letter).lower() for letter in valid_letters if str(letter)}
    if not valid_letters:
        return ""

    checks = [
        # Strict one-token letter responses (A / (A) / A.)
        (r"^\s*\(?([a-z])\)?[\.\)]?\s*$", 1),
        # Explicit declarations where the selected letter is standalone.
        (r"^\s*(?:option|choice|answer)\s*[:\-]?\s*\(?([a-z])\)?[\.\)]?\s*$", 1),
        (r"^\s*(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s*(?:is|:)\s*\(?([a-z])\)?[\.\)]?\s*$", 1),
        (r"^\s*(?:i\s+choose|i\s+pick|choose|pick|select)\s+\(?([a-z])\)?[\.\)]?\s*$", 1),
        # Common explicit forms with short trailing qualifier.
        (r"^\s*(?:option|choice)\s+\(?([a-z])\)?\s+is\s+correct\.?\s*$", 1),
        (r"^\s*(?:answer|the answer)\s+is\s+\(?([a-z])\)?\s*$", 1),
    ]
    for pattern, group_id in checks:
        match = re.search(pattern, predict_raw, flags=re.IGNORECASE)
        if match:
            candidate = match.group(group_id).lower()
            if candidate in valid_letters:
                return candidate
    return ""


def _map_prediction_to_option_letter(prediction: str, option_map: Dict[str, str]) -> str:
    if not option_map:
        return ""

    valid_letters = set(option_map.keys())
    predicted_letter = _extract_predicted_letter(prediction, valid_letters)
    if predicted_letter:
        return predicted_letter

    response = _clean_prediction_text(prediction)
    response_candidates = []
    normalized_response = _normalize_text(response)
    if normalized_response:
        response_candidates.append(normalized_response)
    stripped_response = _normalize_text(_strip_leading_choice_label(response, valid_letters))
    if stripped_response and stripped_response != normalized_response:
        response_candidates.append(stripped_response)

    for candidate in response_candidates:
        matches = [letter for letter, option_text in option_map.items() if candidate == option_text]
        if len(matches) == 1:
            return matches[0]

    for candidate in response_candidates:
        padded_candidate = f" {candidate} "
        contains = [letter for letter, option_text in option_map.items() if f" {option_text} " in padded_candidate]
        if len(contains) == 1:
            return contains[0]

        tokens = set(candidate.split())
        token_matches = [letter for letter, option_text in option_map.items() if " " not in option_text and option_text in tokens]
        if len(token_matches) == 1:
            return token_matches[0]

    return ""


def mmstar_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = _strip_think_prefix(results[0])
    gt = doc["answer"]

    score = exact_match(pred, gt)
    if score == 0.0:
        gt_letter = str(gt).strip().strip("()").lower()
        option_map = _extract_option_map_from_question(doc.get("question", ""))
        if gt_letter in option_map:
            mapped_letter = _map_prediction_to_option_letter(pred, option_map)
            if mapped_letter == gt_letter:
                score = 1.0

    category = doc["category"]
    l2_category = doc["l2_category"]
    return {category: {"question_id": doc["index"], "l2_category": l2_category, "score": score}, "average": {"question_id": doc["index"], "l2_category": l2_category, "score": score}}


def mmstar_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    l2_category_scores = defaultdict(list)
    for result in results:
        score = result["score"]
        l2_category = result["l2_category"]
        l2_category_scores[l2_category].append(score)

    l2_category_avg_score = {}
    for l2_category, scores in l2_category_scores.items():
        avg_score = sum(scores) / len(scores)
        l2_category_avg_score[l2_category] = avg_score
        eval_logger.info(f"{l2_category}: {avg_score:.2f}")

    avg_score = sum(l2_category_avg_score.values()) / len(l2_category_avg_score)
    return avg_score


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <plan>...</plan> or <think>...</think> block if present. If only a closing tag is present, take text after it.
    """
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)
