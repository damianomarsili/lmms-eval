import re

from huggingface_hub import hf_hub_download
from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_hash_answer
from lmms_eval.tasks._task_utils.math_verify_utils import (
    ExprExtractionConfig,
    parse,
    verify,
)
from PIL import Image

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_MATH_EXTRACT_CONFIG = [ExprExtractionConfig(try_extract_without_anchor=True)]


def _extract_answer_tag_content(text):
    match = _ANSWER_TAG_RE.search(str(text))
    if match is None:
        return None
    return match.group(1).strip()


def _parse_math(text):
    return parse(
        str(text),
        extraction_config=_MATH_EXTRACT_CONFIG,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )


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
    return append_hash_answer_instruction(f"{pre_prompt}{question}{post_prompt}")


def geomverse_process_results(doc, results):
    tagged_answer = extract_hash_answer(results[0])
    if tagged_answer == str(results[0]).strip():
        tagged_answer = _extract_answer_tag_content(results[0])
    if tagged_answer is None:
        return {"exact_match": 0.0}
    prediction = _parse_math(tagged_answer)
    target = _parse_math(doc["answer"])
    if len(prediction) == 0 or len(target) == 0:
        return {"exact_match": 0.0}
    return {"exact_match": float(verify(target, prediction, strict=True))}
