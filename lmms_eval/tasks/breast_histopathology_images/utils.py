import re

_YES_NO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def _extract_yes_no(text):
    match = _YES_NO_RE.search(str(text))
    if match is None:
        return None
    return match.group(1).lower()


def breast_histopathology_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def breast_histopathology_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return "Is there a Invasive Ductal Carcinoma shown in the image? Answer yes or no"


def breast_histopathology_doc_to_target(doc):
    return "yes" if int(doc["label"]) == 1 else "no"


def breast_histopathology_process_results(doc, results):
    prediction = _extract_yes_no(results[0])
    target = breast_histopathology_doc_to_target(doc)
    return {"exact_match": float(prediction == target)}
