import re

from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_choice_answer

MEDMNIST_PATH_ID_TO_NAME = {
    0: "adipose",
    1: "background",
    2: "debris",
    3: "lymphocytes",
    4: "mucus",
    5: "smooth muscle",
    6: "normal colon mucosa",
    7: "cancer-associated stroma",
    8: "colorectal adenocarcinoma epithelium",
}

_INDEX_TO_LETTER = {idx: chr(ord("A") + idx) for idx in MEDMNIST_PATH_ID_TO_NAME}
_LETTER_TO_INDEX = {letter: idx for idx, letter in _INDEX_TO_LETTER.items()}
_LETTER_RE = re.compile(r"\b([A-I])\b", flags=re.IGNORECASE)


def _normalize(text):
    return str(text).strip().lower()


def _extract_prediction(text):
    text = str(text).strip()
    letter_match = _LETTER_RE.search(text)
    if letter_match is not None:
        idx = _LETTER_TO_INDEX[letter_match.group(1).upper()]
        return MEDMNIST_PATH_ID_TO_NAME[idx]

    normalized = _normalize(text)
    for _, class_name in MEDMNIST_PATH_ID_TO_NAME.items():
        if normalized == class_name:
            return class_name

    for _, class_name in MEDMNIST_PATH_ID_TO_NAME.items():
        if class_name in normalized:
            return class_name

    return None


def pathmnist_224_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def pathmnist_224_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    options = "\n".join([f"{_INDEX_TO_LETTER[idx]}. {name}" for idx, name in MEDMNIST_PATH_ID_TO_NAME.items()])
    return append_hash_answer_instruction(
        "Which of these options are shown in the image?\n"
        f"{options}\n"
        "Answer with the option letter or exact option text."
    )


def pathmnist_224_doc_to_target(doc):
    label = int(doc["label"][0])
    return MEDMNIST_PATH_ID_TO_NAME[label]


def pathmnist_224_process_results(doc, results):
    prediction = _extract_prediction(extract_choice_answer(results[0], valid_choices="ABCDEFGHI"))
    target = pathmnist_224_doc_to_target(doc)
    return {"exact_match": float(prediction == target)}
