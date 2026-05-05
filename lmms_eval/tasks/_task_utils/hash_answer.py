import re


HASH_ANSWER_INSTRUCTION = 'Output only the final answer after "####".'
_HASH_RE = re.compile(r"####\s*(.+)", flags=re.DOTALL)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_CHOICE_MARKER_RE = re.compile(r"(?i)\b(?:final answer|answer|option|choice)\s*(?:is|:)?\s*(?:\*\*)?\(?([A-Z])\)?(?:\*\*)?\b")
_CHOICE_LINE_RE = re.compile(r"(?i)^\s*(?:[-*]\s*)?(?:\*\*)?\(?([A-Z])\)?(?:\*\*)?(?=\s*$|[\s:.)-])")


def append_hash_answer_instruction(prompt: str) -> str:
    prompt = str(prompt).rstrip()
    if "####" in prompt:
        return prompt
    return f"{prompt}\n{HASH_ANSWER_INSTRUCTION}"


def extract_hash_answer(text: str) -> str:
    text = str(text)
    matches = _HASH_RE.findall(text)
    if not matches:
        return text.strip()
    return matches[-1].strip()


def has_hash_answer(text: str) -> bool:
    return bool(_HASH_RE.search(str(text)))


def extract_choice_answer(text: str, valid_choices: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    answer = extract_hash_answer(text)
    valid = {choice.upper() for choice in valid_choices}

    boxed_matches = [match.strip().upper() for match in _BOXED_RE.findall(answer)]
    boxed_matches = [match for match in boxed_matches if match in valid]
    if boxed_matches:
        return boxed_matches[-1]

    marker_matches = [match.upper() for match in _CHOICE_MARKER_RE.findall(answer)]
    marker_matches = [match for match in marker_matches if match in valid]
    if marker_matches:
        return marker_matches[-1]

    final_line = _final_nonempty_line(answer)
    line_match = _CHOICE_LINE_RE.match(final_line)
    if line_match is not None and line_match.group(1).upper() in valid:
        return line_match.group(1).upper()

    cleaned = re.sub(r"[*_`#>\[\](){}]", " ", answer)
    cleaned = re.sub(r"^\s*[-+]\s*", "", cleaned.strip())
    if len(cleaned.strip()) == 1 and cleaned.strip().upper() in valid:
        return cleaned.strip().upper()
    return answer.strip()


def _final_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[-1]
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
    line = line.strip("*_`#> \t")
    return line


def extract_answer_for_target(text: str, target: str) -> str:
    if has_hash_answer(text):
        return extract_hash_answer(text)

    tail = str(text)[-2000:]
    tag_matches = _ANSWER_TAG_RE.findall(tail)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = _BOXED_RE.findall(tail)
    if boxed_matches:
        return boxed_matches[-1].strip()

    target_text = str(target).strip()
    if target_text.lower() in {"yes", "no"}:
        matches = re.findall(r"\b(yes|no)\b", tail, flags=re.IGNORECASE)
        if matches:
            return matches[-1].lower()

    if _NUMBER_RE.fullmatch(target_text.replace(",", "")) or _NUMBER_RE.fullmatch(target_text):
        matches = _NUMBER_RE.findall(tail)
        if matches:
            return matches[-1].replace(",", "")

    answer_markers = re.findall(
        r"(?i)(?:final answer|answer|therefore|thus|so)\s*(?:is|:)?\s*([^\n.]+)",
        tail,
    )
    if answer_markers:
        return answer_markers[-1].strip("*_`#> \t")

    return _final_nonempty_line(tail)
