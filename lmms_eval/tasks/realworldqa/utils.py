import re

from lmms_eval.filters.extraction import ExtendedRegexFilter
from lmms_eval.filters.transformation import MapFilter

REPLACE_PROMPT = "Please answer directly with only the letter of the correct option and nothing else."


def realworldqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def realworldqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    question = doc["question"].strip()
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        question = question.replace(REPLACE_PROMPT, "")
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


# number_words_to_digits = {
#     "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
#     "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
#     "ten": "10"
# }


def realworldqa_process_results(doc, results):
    """
    Score answers with awareness of the target type:
    - If the target is a single letter (e.g., multiple choice), compare only the first letter from the model output.
    - Otherwise, fall back to an exact (lowercased, trimmed) comparison.
    """
    if "</think>" in results[0]:
        raw_pred = _strip_think_prefix(results[0]).lower().strip()
    else:
        raw_pred = results[0].lower().strip()

    target_raw = str(doc["answer"]).strip()
    target = target_raw.lower()

    # Yes/No targets: normalize both sides
    if target in {"yes", "no"}:
        pred_norm = _normalize_yes_no(raw_pred)
        score = 1.0 if pred_norm == target else 0.0
        return {"exact_match": score}

    # Numeric targets: pull the last number from the prediction and compare; fall back to direct string compare
    if target.isdigit():
        gt_num = int(target)
        pred_num = int(raw_pred)

        score = 1.0 if pred_num == gt_num else 0.0
        return {"exact_match": score}

    # Single-letter alpha targets: use the last standalone alpha token
    if len(target) == 1 and target.isalpha():
        score = 1.0 if raw_pred == target else 0.0
        return {"exact_match": score}


def _strip_think_prefix(text: str) -> str:
    """
    Drop a leading <think>...</think> block if present.
    If only a closing </think> is present, take text after it.
    """
    if not isinstance(text, str):
        return text
    if "</think>" in text.lower():
        return text.split("</think>")[-1].strip()
    return re.sub(r"(?is)^\s*<think>.*?</think>\s*", "", text, count=1)


def _normalize_yes_no(text: str) -> str:
    # Look for explicit yes/no tokens
    match = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if match:
        return match[-1].lower()
    token = re.sub(r"[^a-z0-9]+", "", text.lower())
    if token in {"yes", "y", "yeah", "yep", "true", "1"}:
        return "yes"
    if token in {"no", "n", "nope", "false", "0"}:
        return "no"
    parts = text.lower().split()
    return parts[-1] if parts else ""


class NumberWordsToDigitsFilter(MapFilter):
    def __init__(self) -> None:
        mapping_dict = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        super().__init__(mapping_dict, default_value=None)

    def apply(self, resps, docs):
        def filter_set(inst):
            return [self.mapping_dict.get(resp.lower(), resp) for resp in inst]

        return [filter_set(resp) for resp in resps]


class MultiChoiceRegexFilter(ExtendedRegexFilter):
    def __init__(self, *args, **kwargs):
        """
        regex_pattern: The basic regex pattern to use. If fails to match, we will use the customized match procedure
                        - step 1 : We parse the choices between ([A-Z])s then try to find these choices in the response.
                        - step 2 : We parse the choice with regex :[\s]*([A-?]), where ? varies by number of choices.
        group_select: Selects the (group_select)th match from the findall result.
        ignore_case: Ignores the case during step 1 matching
        ignore_punctuation: Remove the punctuation during step 1 matching
        regexes_to_ignore: Remove these regexes during step 1 matching
        """
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        # here, we assume we have a list, in which each element is
        # a list of model responses for some particular input/target pair.
        # so we process each of these (same input/target response sets)
        # independently (and keep them a list.)

        filtered_resps = []

        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = "A"

            without_paren_fallback_regexes = []
            without_paren_to_target = {}

            # Regex to extract multiple choice options from the question
            multiple_choices_regex = re.compile(r"\b([A-Z])\.\s+([^\n]*)")
            matches = multiple_choices_regex.findall(doc["question"])

            has_choices = len(matches) > 0

            # Build regex patterns and mappings for each choice
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f"{re.escape(choice_text)}")
                choice_to_alpha[choice_text] = next_alpha

                next_alpha = chr(ord(next_alpha) + 1)

            # Compile regex to match any of the extracted choices
            fallback_regex = re.compile("|".join(fallback_regexes)) if fallback_regexes else None

            # Process each response
            filtered = []
            for resp in r:
                # If the question has no explicit choices, keep the response as-is (trimmed)
                if not has_choices:
                    filtered.append(resp.strip())
                    continue

                # Fast path: if a standalone choice letter (A-D/a-d) appears, use it directly
                letter_match = re.search(r"\b([A-D])\b", resp, flags=re.IGNORECASE)
                if letter_match:
                    filtered.append(letter_match.group(1).upper())
                    continue

                # Remove any punctuation and extra spaces
                cleaned_resp = re.sub(r"[^\w\s]", "", resp).strip()
                if fallback_regex:
                    match = fallback_regex.search(cleaned_resp)
                else:
                    match = None

                if match and match.group() in choice_to_alpha:
                    # Map the matched choice text back to its corresponding letter
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    # If no match, return the cleaned response untouched (no first-letter forcing)
                    filtered.append(cleaned_resp)

            filtered_resps.append(filtered[0])

        return filtered_resps
