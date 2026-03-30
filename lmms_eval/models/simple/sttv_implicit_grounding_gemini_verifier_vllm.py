import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_implicit_grounding_vllm import STTVImplicitGroundingVLLM

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from training.gemini_objectives import (  # noqa: E402
    build_gemini_runtime_config,
    generate_gemini_logic_teacher_judgment,
    load_gemini_prompt_template,
)


@register_model("sttv_implicit_grounding_gemini_verifier_vllm")
class STTVImplicitGroundingGeminiVerifierVLLM(STTVImplicitGroundingVLLM):
    """
    Eval-time implicit-grounding flow with Gemini as logic verifier:
    query -> <reason><answer> -> Gemini edits -> rewritten <reason><answer>.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        generation_chunk_max_new_tokens: int = 768,
        logic_verifier_rounds: int = 1,
        logic_verifier_max_new_tokens: int = 128,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        min_image_pixels: int = 28,
        seed: int = 1,
        disable_log_stats: bool = False,
        gemini_model: str = "gemini-3-flash-preview",
        gemini_api_key: Optional[str] = None,
        gemini_vertexai: Union[bool, str, int, float] = False,
        gemini_vertex_project: Optional[str] = None,
        gemini_vertex_location: Optional[str] = None,
        gemini_service_account_file: Optional[str] = None,
        gemini_http_api_version: Optional[str] = None,
        gemini_timeout_s: int = 180,
        gemini_max_retries: int = 5,
        gemini_retry_sleep_s: float = 5.0,
        gemini_max_output_tokens: int = 512,
        gemini_max_image_side: int = 768,
        gemini_global_max_inflight: int = 1,
        gemini_global_slot_wait_s: float = 900.0,
        gemini_global_slot_dir: str = "/tmp/sttv_gemini_slots",
        gemini_failure_mode: str = "zero",
        gemini_debug_print_io: Union[bool, str, int, float] = False,
        gemini_logic_teacher_prompt_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            batch_size=batch_size,
            depth=depth,
            prompt_path=prompt_path,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            generation_chunk_max_new_tokens=generation_chunk_max_new_tokens,
            logic_verifier_rounds=logic_verifier_rounds,
            logic_verifier_max_new_tokens=logic_verifier_max_new_tokens,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )
        runtime_kwargs = {
            "gemini_model": gemini_model,
            "gemini_api_key": gemini_api_key,
            "gemini_vertexai": self._coerce_bool(gemini_vertexai),
            "gemini_vertex_project": gemini_vertex_project,
            "gemini_vertex_location": gemini_vertex_location,
            "gemini_service_account_file": gemini_service_account_file,
            "gemini_http_api_version": gemini_http_api_version,
            "gemini_timeout_s": gemini_timeout_s,
            "gemini_max_retries": gemini_max_retries,
            "gemini_retry_sleep_s": gemini_retry_sleep_s,
            "gemini_max_output_tokens": gemini_max_output_tokens,
            "gemini_max_image_side": gemini_max_image_side,
            "gemini_global_max_inflight": gemini_global_max_inflight,
            "gemini_global_slot_wait_s": gemini_global_slot_wait_s,
            "gemini_global_slot_dir": gemini_global_slot_dir,
            "gemini_failure_mode": gemini_failure_mode,
            "gemini_debug_print_io": self._coerce_bool(gemini_debug_print_io),
        }
        self.gemini_runtime_config = build_gemini_runtime_config(runtime_kwargs)
        self.gemini_logic_teacher_prompt_template = load_gemini_prompt_template(
            gemini_logic_teacher_prompt_path,
            default_filename="gemini_logic_teacher_judge_implicit_grounding_instructions.txt",
        )

    def _request_gemini_logic_teacher_judgment(
        self,
        query: str,
        current_answer_output: str,
        visuals: List[object],
    ) -> Dict[str, object]:
        images: List[Image.Image] = []
        for item in visuals:
            if isinstance(item, Image.Image):
                images.append(item.convert("RGB"))
        return generate_gemini_logic_teacher_judgment(
            config=self.gemini_runtime_config,
            prompt_template=self.gemini_logic_teacher_prompt_template,
            query=query,
            detected_objects="",
            current_answer=current_answer_output,
            proposed_self_edits="",
            images=images,
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(x[0]), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        self.last_generation_metadata = None
        all_generation_metadata: List[Dict[str, object]] = []
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                sample_task = task[i]
                sample_split = split[i]
                sample_doc_id = doc_id[i]
                sample_doc_to_visual = doc_to_visual[i]
                sample_visual = sample_doc_to_visual(self.task_dict[sample_task][sample_split][sample_doc_id])
                sample_gen_kwargs = dict(all_gen_kwargs[i])

                if "<image" in context:
                    context = re.sub(r"<image\s*\d+>", "", context)
                    context = context.replace("<image>", "")

                visuals = sample_visual if isinstance(sample_visual, list) else [sample_visual]
                visuals = [item for item in visuals if item is not None]

                initial_prompt = self._build_initial_answer_prompt(context)
                initial_messages = self._build_messages(initial_prompt, visuals)
                current_answer_output, _ = self._generate_once(
                    initial_messages,
                    sample_gen_kwargs,
                    stop_sequences=["</answer>"],
                    max_new_tokens=max(1, self.generation_chunk_max_new_tokens),
                )
                generation_metadata: Dict[str, object] = {
                    "requested_logic_verifier_rounds": int(self.logic_verifier_rounds),
                    "performed_logic_verifier_rounds": 0,
                    "initial_answer_draft": current_answer_output,
                    "logic_verifier_rounds": [],
                    "logic_feedback_source": "gemini_teacher",
                }
                logic_round_records = generation_metadata["logic_verifier_rounds"]
                assert isinstance(logic_round_records, list)

                for round_index in range(1, self.logic_verifier_rounds + 1):
                    round_record: Dict[str, object] = {
                        "round_index": round_index,
                        "answer_before_feedback": current_answer_output,
                        "feedback_source": "gemini_teacher",
                    }
                    t0 = time.perf_counter()
                    teacher_judgment = self._request_gemini_logic_teacher_judgment(
                        query=context,
                        current_answer_output=current_answer_output,
                        visuals=visuals,
                    )
                    round_record["gemini_time_s"] = max(0.0, time.perf_counter() - t0)
                    teacher_edits_raw = teacher_judgment.get("teacher_edits", [])
                    if isinstance(teacher_edits_raw, str):
                        teacher_edits_list = [line.strip() for line in teacher_edits_raw.splitlines() if line.strip()]
                    elif isinstance(teacher_edits_raw, (list, tuple)):
                        teacher_edits_list = [str(line or "").strip() for line in teacher_edits_raw if str(line or "").strip()]
                    else:
                        teacher_edits_list = []
                    raw_feedback_text = "\n".join(teacher_edits_list).strip()
                    round_record["self_verifier_feedback_raw"] = raw_feedback_text
                    round_record["gemini_teacher_edits_raw"] = teacher_edits_list
                    round_record["gemini_teacher_response_raw"] = str(teacher_judgment.get("raw_text", "") or "")
                    round_record["gemini_current_answer_score"] = float(
                        teacher_judgment.get("current_answer_score", 0.0) or 0.0
                    )
                    round_record["gemini_self_edit_score"] = float(teacher_judgment.get("self_edit_score", 0.0) or 0.0)
                    round_record["gemini_self_edit_reason"] = str(teacher_judgment.get("self_edit_reason", "") or "")
                    round_record["gemini_failed"] = bool(teacher_judgment.get("failed", False))
                    round_record["gemini_error"] = str(teacher_judgment.get("error", "") or "")

                    logic_feedback, logic_parse_valid = self._parse_logic_step_edits_optional(
                        raw_feedback_text, current_answer_output
                    )
                    round_record["self_verifier_feedback_parse_valid"] = bool(logic_parse_valid)
                    round_record["self_verifier_feedback_parsed"] = logic_feedback

                    fallback_used = False
                    if (not logic_parse_valid) or (not str(logic_feedback or "").strip()):
                        logic_feedback = "No valid verifier feedback was produced. Re-emit the current answer unchanged."
                        fallback_used = True
                    round_record["self_verifier_feedback_fallback_used"] = fallback_used

                    rewrite_prompt = self._build_answer_rewrite_prompt(context, current_answer_output, logic_feedback)
                    rewrite_messages = self._build_messages(rewrite_prompt, visuals)
                    current_answer_output, _ = self._generate_once(
                        rewrite_messages,
                        sample_gen_kwargs,
                        stop_sequences=["</answer>"],
                        max_new_tokens=max(1, self.generation_chunk_max_new_tokens),
                    )
                    round_record["answer_after_rewrite"] = current_answer_output
                    logic_round_records.append(round_record)

                generation_metadata["performed_logic_verifier_rounds"] = len(logic_round_records)
                generation_metadata["final_answer_output"] = current_answer_output

                res.append(current_answer_output)
                all_generation_metadata.append(generation_metadata)
                self.cache_hook.add_partial("generate_until", (context, sample_gen_kwargs), current_answer_output)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        self.last_generation_metadata = re_ords.get_original(all_generation_metadata)
        pbar.close()
        return res
