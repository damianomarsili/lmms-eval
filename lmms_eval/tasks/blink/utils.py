import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

    config = yaml.safe_load("".join(safe_data))


_OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_TEXT_FALLBACK_TASK_KEYS = {"counting", "spatial_relation", "blink_counting", "blink_spatial_relation"}


def _option_letters(num_options: int) -> List[str]:
    return list(_OPTION_LABELS[:num_options])


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = _strip_think_prefix(text)
    text = text.strip()

    # Collect all standalone A-E letter hits and return the last one
    hits = re.findall(r"(?i)(?<![A-Z])([A-E])(?![A-Z])", text)
    if hits:
        return hits[-1].upper()

    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <plan>...</plan> or <think>...</think> block if present and return the remainder.
    If only a closing tag remains (e.g., upstream stripped the opener), take text after the last closing tag.
    """
    if not isinstance(text, str):
        return text
    if re.search(r"</(?:plan|think)>", text, flags=re.IGNORECASE):
        return re.split(r"</(?:plan|think)>", text, flags=re.IGNORECASE)[-1].strip()
    return re.sub(r"(?is)^\s*<(?:plan|think)>.*?</(?:plan|think)>\s*", "", text, count=1)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_leading_choice_label(text: str, valid_letters: Sequence[str]) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    letters = "".join(sorted({str(letter).upper() for letter in valid_letters if str(letter)}))
    if not letters:
        return cleaned

    patterns = [
        rf"^\(\s*[{letters}]\s*\)\s*",
        rf"^\[\s*[{letters}]\s*\]\s*",
        rf"^[{letters}]\s*[\.\):\-]\s*",
        rf"^[{letters}]\s+",
    ]
    for pattern in patterns:
        updated = re.sub(pattern, "", cleaned, count=1, flags=re.IGNORECASE)
        if updated != cleaned:
            return updated.strip()
    return cleaned


def _is_text_fallback_task(doc: Dict[str, Any]) -> bool:
    task_candidates = {
        str(doc.get("sub_task", "")).strip().lower(),
        str(doc.get("dataset_name", "")).strip().lower(),
        str(doc.get("task", "")).strip().lower(),
    }
    normalized_task_candidates = {task.replace(" ", "_") for task in task_candidates if task}
    return bool(normalized_task_candidates & _TEXT_FALLBACK_TASK_KEYS)


def _map_response_to_choice_letter(response: str, choices: Sequence[Any]) -> str:
    if not response or not choices:
        return ""

    letters = _option_letters(len(choices))
    if not letters:
        return ""

    choice_aliases: Dict[str, set[str]] = {}
    for letter, choice in zip(letters, choices):
        raw_choice = str(choice).strip()
        if not raw_choice:
            continue
        stripped_choice = _strip_leading_choice_label(raw_choice, letters)
        aliases = {_normalize_text(raw_choice), _normalize_text(stripped_choice)}
        aliases.discard("")
        if aliases:
            choice_aliases[letter] = aliases

    if not choice_aliases:
        return ""

    response_candidates: List[str] = []
    normalized_response = _normalize_text(response)
    if normalized_response:
        response_candidates.append(normalized_response)
    stripped_response = _normalize_text(_strip_leading_choice_label(response, letters))
    if stripped_response and stripped_response != normalized_response:
        response_candidates.append(stripped_response)

    for candidate in response_candidates:
        exact_matches = [letter for letter, aliases in choice_aliases.items() if candidate in aliases]
        if len(exact_matches) == 1:
            return exact_matches[0]

    for candidate in response_candidates:
        padded_candidate = f" {candidate} "
        contains_matches = []
        for letter, aliases in choice_aliases.items():
            if any(f" {alias} " in padded_candidate for alias in aliases):
                contains_matches.append(letter)
        if len(contains_matches) == 1:
            return contains_matches[0]

        tokens = set(candidate.split())
        token_matches = []
        for letter, aliases in choice_aliases.items():
            if any(" " not in alias and alias in tokens for alias in aliases):
                token_matches.append(letter)
        if len(token_matches) == 1:
            return token_matches[0]

    return ""


def blink_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    options_labels = ["A", "B", "C", "D", "E"]
    num_options = len(doc["choices"])
    options_current_task = ", ".join(options_labels[:num_options])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(options_current_task) + doc["prompt"]
    return prompt


def blink_doc_to_visual(doc: dict) -> list:
    keys = doc.keys()
    image_keys = [item for item in keys if re.match(r"^image_\d+$", item)]
    image_list = []
    for image_key in image_keys:
        image = doc[image_key]
        if image is not None:
            image_list.append(image.convert("RGB"))
    return image_list


def blink_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "blink_acc"
    # extract grounded answer
    grounded_output = str(doc["answer"]).strip("()").strip().upper()
    response = _strip_think_prefix(result[0])
    choices = doc.get("choices", [])
    valid_letters = set(_option_letters(len(choices)))

    # extract predicted answer
    pred_letter = _extract_answer_letter(response)
    if valid_letters and pred_letter not in valid_letters:
        pred_letter = ""

    # Keep traditional letter-based scoring first, then fall back to option-content
    # matching for BLINK subsets where models often output values like "yes"/"no"/"1".
    if pred_letter != grounded_output and _is_text_fallback_task(doc):
        mapped_letter = _map_response_to_choice_letter(response, choices)
        if mapped_letter:
            pred_letter = mapped_letter

    flag = pred_letter == grounded_output

    omnispatial_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "sub_task": doc["sub_task"], "is_correct": flag}
    return {key_name: omnispatial_submission}


def blink_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
