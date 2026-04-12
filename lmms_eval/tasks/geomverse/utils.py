import re

from huggingface_hub import hf_hub_download
from PIL import Image

_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)


def _extract_number(text):
    match = _NUMBER_RE.search(str(text).replace(",", ""))
    if match is None:
        return None
    return float(match.group(0))


def _extract_answer_tag_content(text):
    match = _ANSWER_TAG_RE.search(str(text))
    if match is None:
        return None
    return match.group(1).strip()


def geomverse_doc_to_visual(doc):
    image_path = hf_hub_download(
        repo_id="LibraTree/geomverse",
        repo_type="dataset",
        filename=doc["image"],
    )
    return [Image.open(image_path).convert("RGB")]


def geomverse_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["problem_text"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def geomverse_process_results(doc, results):
    tagged_answer = _extract_answer_tag_content(results[0])
    prediction = _extract_number(tagged_answer) if tagged_answer is not None else None
    target = float(doc["answer"])
    if prediction is None:
        return {"exact_match": 0.0}
    return {"exact_match": float(abs(prediction - target) < 1e-4)}
