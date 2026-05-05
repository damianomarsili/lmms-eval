from datasets import Dataset

from lmms_eval.tasks.charxiv.constant import REASONING_RESP_INST
from lmms_eval.tasks.charxiv.descriptive_utils import descriptive_query_helper
from lmms_eval.tasks.charxiv.reasoning_utils import get_number_instruction
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from lmms_eval.tasks._task_utils.hash_answer import append_hash_answer_instruction, extract_hash_answer

_ANSWER_PROCESSOR = EvalAIAnswerProcessor()


def _normalize_answer(text):
    return _ANSWER_PROCESSOR(str(text).strip())


def charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    inst_category = doc["reasoning_q_source"]
    if inst_category in [1, 2, 3]:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"])
    # 4: number-in-general -> need to specify the number of decimal places
    elif inst_category == 4:
        question = REASONING_RESP_INST[inst_category].format(doc["reasoning_q"], get_number_instruction(doc["reasoning_a"]))
    return append_hash_answer_instruction(question)


def charxiv_descriptive_process_docs(dataset: Dataset) -> Dataset:
    # Four descriptive questions for each reasoning question
    dataset = dataset.repeat(4)

    def _process_row(example, indice):
        q_number = indice % 4 + 1
        descriptive_q = example[f"descriptive_q{q_number}"]
        qid = descriptive_q
        subplot_loc = example["subplot_loc"]
        if subplot_loc is None:
            subplot_row = example["subplot_row"]
            subplot_col = example["subplot_col"]
            subplot_loc = [subplot_row, subplot_col]
        descriptive_q = descriptive_query_helper(descriptive_q, subplot_loc)
        example[f"descriptive_q"] = descriptive_q
        example[f"descriptive_a"] = example[f"descriptive_a{q_number}"]
        return {"qid": qid, **example}

    # Keep preprocessing in-process to avoid multiprocessing pickling issues
    # with task module state (e.g., API clients carrying SSL contexts).
    dataset = dataset.map(_process_row, with_indices=True)
    return dataset


def charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return append_hash_answer_instruction(doc["descriptive_q"])


def charxiv_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def charxiv_descriptive_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_descriptive_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages


def charxiv_reasoning_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = charxiv_reasoning_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = charxiv_doc_to_visual(doc)
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    return messages


def charxiv_descriptive_process_results(doc, results):
    prediction = _normalize_answer(extract_hash_answer(results[0]))
    target = _normalize_answer(doc["descriptive_a"])
    return {"exact_match": float(prediction == target)}


def charxiv_reasoning_process_results(doc, results):
    prediction = _normalize_answer(extract_hash_answer(results[0]))
    target = _normalize_answer(doc["reasoning_a"])
    return {"exact_match": float(prediction == target)}
