import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server
from lmms_eval.tasks._task_utils.math_verify_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)

try:
    from lmms_eval.tasks.mathvision.eval_utils import (
        find_math_answer,
        is_equal,
        is_number,
    )
except ImportError as e:
    eval_logger.warning(f"Error importing eval_utils from lmms_eval.tasks.mathvision.eval_utils: {e}")
    pass

NUM_SECONDS_TO_SLEEP = 5
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.IGNORECASE | re.DOTALL)
_MATH_NORM_CONFIG = NormalizationConfig(
    basic_latex=True,
    units=True,
    malformed_operators=True,
    nits=True,
    boxed="all",
    equations=False,
)
_MATH_EXTRACT_CONFIG = [
    LatexExtractionConfig(boxed_match_priority=0, normalization_config=_MATH_NORM_CONFIG),
    ExprExtractionConfig(try_extract_without_anchor=True),
]

# Initialize the judge server
API_TYPE = os.getenv("API_TYPE", "openai")
GPT_MODEL = os.getenv("MODEL_VERSION", "gpt-4o-2024-11-20")

server_config = ServerConfig(
    model_name=GPT_MODEL,
)
server = get_server(server_name=API_TYPE, config=server_config)


def mathvision_doc_to_visual(doc):
    return [doc["decoded_image"].convert("RGB")]


def mathvision_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question, choices = doc["question"], doc["options"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None:
        mc_prompt = "\n" + lmms_eval_specific_kwargs["mc_prompt"]

    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}" + mc_prompt
    else:
        query_prompt += question
    return query_prompt


def mathvision_gpt_eval_process_results(doc, results):
    correct_list = []
    for pred in results:
        model_answer = pred.strip()
        gt_answer = str(doc["answer"])
        question = doc["question"]

        try:
            # Use the llm_judge API for binary evaluation
            result = server.evaluate_binary(question=question, answer=gt_answer, prediction=model_answer, output_format="0/1")

            # Parse the result
            if result["success"]:
                judge_response = result["result"]
                correct_list.append(judge_response)
            else:
                eval_logger.error(f"Judge evaluation failed: {result.get('raw_response', 'Unknown error')}")
                correct_list.append(False)

        except Exception as e:
            eval_logger.error(f"Error getting judge response: {e}")
            correct_list.append(False)

    # Calculate the average score for this document
    avg_score = sum(1 if score else 0 for score in correct_list) / len(correct_list) if correct_list else 0
    return {"llm_as_judge_eval": avg_score}


def mathvision_process_results(doc, results):
    def extract_answer_tag_content(text):
        match = _ANSWER_TAG_RE.search(str(text))
        if match is None:
            return None
        return match.group(1).strip()

    def parse_math(text):
        return parse(
            str(text),
            extraction_config=_MATH_EXTRACT_CONFIG,
            extraction_mode="any_match",
            fallback_mode="first_match",
        )

    def parse_choice(text, options):
        option_letters = tuple(chr(ord("A") + i) for i in range(len(options)))
        return parse(
            str(text),
            extraction_config=[StringExtractionConfig(strings=option_letters, try_extract_without_anchor=True, lowercase=False)],
            extraction_mode="first_match",
            fallback_mode="no_fallback",
        )

    correct_list = []
    for pred in results:
        prediction_text = extract_answer_tag_content(pred)
        if prediction_text is None:
            prediction_text = pred.strip()

        gt_answer = str(doc["answer"]).strip()
        if len(doc["options"]) > 0:
            pred_parsed = parse_choice(prediction_text, doc["options"])
            gold_parsed = parse_choice(gt_answer, doc["options"])
        else:
            pred_parsed = parse_math(prediction_text)
            gold_parsed = parse_math(gt_answer)

        correct = len(pred_parsed) > 0 and len(gold_parsed) > 0 and verify(gold_parsed, pred_parsed, strict=True)
        correct_list.append(correct)
    return {
        "mathvision_standard_eval": {
            # "question": doc["question"],
            # "answer": doc["answer"],
            "response": results,
            # "subject": doc["subject"],
            # "level": doc["level"],
            "scores": correct_list,
        },
    }


def mathvision_aggregate_results_eval(results):
    total = len(results)
    correct = sum(1 for idx, result in enumerate(results) if results[idx]["scores"][0])
    accuracy = round(correct / total * 100, 2)
    return accuracy
