from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def vqa_rad_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def vqa_rad_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def vqa_rad_process_results(doc, results):
    prediction = _ANSWER_PROCESSOR(results[0].strip())
    target = _ANSWER_PROCESSOR(doc["answer"])
    return {"exact_match": float(prediction == target)}
