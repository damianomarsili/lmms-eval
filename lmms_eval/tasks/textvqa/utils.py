import datetime
import json
import statistics
import re

from loguru import logger as eval_logger
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


# --- helpers --------------------------------------------------------------

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000
}

def _words_to_number(s: str):
    """
    Convert simple English number words to an integer (supports up to thousands).
    Returns None if not convertible.
    """
    s = re.sub(r"[^a-z\s\-]", "", s.lower()).strip()
    if not s:
        return None
    parts = re.split(r"[\s\-]+", s)

    total = 0
    current = 0
    matched_any = False
    for w in parts:
        if w not in _NUM_WORDS:
            return None
        matched_any = True
        val = _NUM_WORDS[w]
        if val in (100, 1000):
            if current == 0:
                current = 1
            current *= val
            if val == 1000:
                total += current
                current = 0
        else:
            current += val
    if not matched_any:
        return None
    return total + current

def _to_number(s: str):
    """
    Try to parse a string as a number (digit or word form).
    Returns an int if possible, else a float if possible, else None.
    """
    if not isinstance(s, str):
        return None
    s_clean = s.strip().lower()

    # Try integer/float with punctuation removed (e.g., "1,000" -> "1000")
    num_like = re.sub(r"[,\s]", "", s_clean)
    # allow leading/trailing punctuation like '.' or '%'
    num_like = re.sub(r"[^\d\.\-+eE]", "", num_like)
    try:
        if re.fullmatch(r"[+-]?\d+", num_like):
            return int(num_like)
        return float(num_like)
    except Exception:
        pass

    # Try words ("ten", "twenty one")
    wnum = _words_to_number(s_clean)
    if wnum is not None:
        return int(wnum)

    return None

def _answers_equivalent(a: str, b: str) -> bool:
    """
    Equality after EvalAI normalization, plus numeric equivalence.
    """
    if a == b:
        return True
    na, nb = _to_number(a), _to_number(b)
    if na is not None and nb is not None:
        return float(na) == float(nb)
    return False


# --- task functions ------------------------------------------------------

def textvqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def textvqa_process_results(doc, result):
    """
    TextVQA scoring with:
      - EvalAI normalization (as before)
      - Numeric equivalence (e.g., '10' == 'ten')
      - VQA consensus when multiple GT answers exist
    """
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."

    # Normalize model prediction
    resAns = str(eval_ai_processor(result[0]))

    # Build normalized GT list without mutating `doc`
    gt_list = []
    if doc.get("answers"):
        gt_list = [str(eval_ai_processor(a)) for a in doc["answers"]]
    elif doc.get("answer"):
        gt_list = [str(eval_ai_processor(doc["answer"]))]

    # Compute accuracy
    if not gt_list:
        accuracy = 0.0
    elif len(gt_list) == 1:
        accuracy = 1.0 if _answers_equivalent(gt_list[0], resAns) else 0.0
    else:
        gtAcc = []
        n = len(gt_list)
        for i in range(n):
            otherGT = [gt_list[j] for j in range(n) if j != i]
            matches = sum(1 for g in otherGT if _answers_equivalent(g, resAns))
            gtAcc.append(min(1.0, matches / 3.0))
        accuracy = statistics.mean(gtAcc)

    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }


def textvqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    pre_prompt = ""
    post_prompt = ""
    ocr_ref = ""

    if lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
        post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
        if lmms_eval_specific_kwargs.get("ocr"):
            ocr_tokens = doc.get("ocr_tokens") or []
            if ocr_tokens:
                ocr_ref = f"\nReference OCR token: {', '.join(ocr_tokens)}"

    question = (doc.get("question") or "").capitalize()
    return f"{pre_prompt}{question}{ocr_ref}{post_prompt}"


def textvqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    path = generate_submission_file(f"textvqa_submission_{now_date_time}.json", args)
    with open(path, "w") as f:
        json.dump(results, f)
    eval_logger.info(f"Submission file saved to {path}")
