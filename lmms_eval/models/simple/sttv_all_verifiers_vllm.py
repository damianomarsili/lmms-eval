import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_vllm import LocEntry, STTVVLLM

LOGIC_STATUS_PATTERN = re.compile(r"(?im)^\s*STATUS\s*:\s*(KEEP|REVISE)\s*$")
LOGIC_FEEDBACK_PATTERN = re.compile(r"(?ims)^\s*FEEDBACK\s*:\s*(.*?)\s*$")
LOGIC_SPATIAL_PATTERN = re.compile(r"(?im)^\s*SPATIAL_CORRECTNESS\s*:\s*(correct|incorrect)\s*$")
LOGIC_ATTRIBUTE_PATTERN = re.compile(r"(?im)^\s*ATTRIBUTE_CORRECTNESS\s*:\s*(correct|incorrect)\s*$")
LOGIC_REASONING_PATTERN = re.compile(r"(?im)^\s*LOGIC_CORRECTNESS\s*:\s*(correct|incorrect)\s*$")


@register_model("sttv_all_verifiers_vllm")
class STTVAllVerifiersVLLM(STTVVLLM):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        prompt_path: Optional[str] = None,
        verifier_prompt_path: Optional[str] = None,
        logic_self_verifier_prompt_path: Optional[str] = None,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        loc_verifier_rounds: int = 3,
        verifier_max_new_tokens: int = 96,
        verifier_image_side: int = 1024,
        logic_verifier_rounds: int = 2,
        logic_verifier_max_new_tokens: int = 96,
        generation_max_new_tokens: int = 2048,
        generation_chunk_max_new_tokens: int = 256,
        trust_remote_code: Optional[bool] = True,
        chat_template: Optional[str] = None,
        min_image_pixels: int = 28,
        seed: int = 1,
        disable_log_stats: bool = False,
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
            verifier_prompt_path=verifier_prompt_path,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            loc_verifier_rounds=loc_verifier_rounds,
            verifier_max_new_tokens=verifier_max_new_tokens,
            verifier_image_side=verifier_image_side,
            generation_max_new_tokens=generation_max_new_tokens,
            generation_chunk_max_new_tokens=generation_chunk_max_new_tokens,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )
        self.logic_verifier_rounds = max(0, int(logic_verifier_rounds))
        self.logic_verifier_max_new_tokens = int(logic_verifier_max_new_tokens)
        self.logic_self_verifier_template = self._load_logic_self_verifier_template(logic_self_verifier_prompt_path)

    def _load_logic_self_verifier_template(self, prompt_path: Optional[str]) -> str:
        if prompt_path is None:
            prompt_file = Path(__file__).resolve().parents[3] / "prompts" / "logic_self_verifier_instructions_all_verifiers.txt"
        else:
            prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Logic self-verifier prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Logic self-verifier prompt file is empty: {prompt_file}")
        return prompt_text

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str) -> str:
        query_text = query.strip()
        return (
            f"Original query:\n{query_text}\n\n"
            "Detected objects (in [x_min, y_min, x_max, y_max] format with coordinates in [0,1000]):\n"
            f"{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags and then putting ONLY your final "
            "answer inside <answer>. Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    def _build_logic_self_verifier_prompt(self, query: str, latest_answer_output: str) -> str:
        return self.logic_self_verifier_template.format(
            query=query.strip(),
            answer=str(latest_answer_output or "").strip(),
        )

    def _parse_logic_self_verifier_output(self, text: str) -> Tuple[str, str, bool]:
        raw = str(text or "").strip()
        spatial_match = LOGIC_SPATIAL_PATTERN.search(raw)
        attribute_match = LOGIC_ATTRIBUTE_PATTERN.search(raw)
        reasoning_match = LOGIC_REASONING_PATTERN.search(raw)
        status_match = LOGIC_STATUS_PATTERN.search(raw)
        feedback_match = LOGIC_FEEDBACK_PATTERN.search(raw)

        feedback = feedback_match.group(1).strip() if feedback_match else ""

        if spatial_match is None or attribute_match is None or reasoning_match is None:
            return "INVALID", "", False

        rubric_values = [
            spatial_match.group(1).strip().lower(),
            attribute_match.group(1).strip().lower(),
            reasoning_match.group(1).strip().lower(),
        ]
        has_incorrect = any(value == "incorrect" for value in rubric_values)
        derived_status = "REVISE" if has_incorrect else "KEEP"

        # Accept explicit STATUS if present, but enforce rubric-driven behavior.
        if status_match is not None:
            parsed_status = status_match.group(1).upper().strip()
            if parsed_status not in {"KEEP", "REVISE"}:
                return "INVALID", "", False
        if has_incorrect and not feedback:
            return "INVALID", "", False
        if derived_status == "KEEP" and not feedback:
            feedback = "The current <reason>/<answer> is correct. Re-emit unchanged."
        return derived_status, feedback, True

    def _build_answer_rewrite_prompt(
        self,
        query: str,
        latest_bbox_block: str,
        current_answer_output: str,
        logic_feedback: str,
    ) -> str:
        clean_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        return (
            f"{clean_prompt}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            f"Feedback: {logic_feedback}\n\n"
            "Please output exactly one full <reason> block and then one full <answer> block. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else. "
            "Do not output any <bbox_2d>. "
            "If feedback indicates the current answer is already correct, re-emit the current answer draft unchanged."
        )

    def _generate_with_verifier(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        visuals: List[Image.Image],
        query: str,
    ) -> str:
        output_chunks: List[str] = []
        max_new_tokens_per_chunk = self.generation_chunk_max_new_tokens
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'
        total_generated_tokens = 0
        max_total_tokens = self.generation_max_new_tokens
        if max_total_tokens <= 0:
            return ""

        remaining = max_total_tokens - total_generated_tokens
        chunk, new_tokens = self._generate_once(
            messages,
            gen_kwargs,
            stop_sequences=["</bbox_2d>"],
            max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
        )
        total_generated_tokens += new_tokens
        if not chunk.strip():
            return ""
        output_chunks.append(chunk)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

        loc_payloads = self._extract_bbox_2d_payloads(chunk)
        if len(loc_payloads) != 1:
            return "".join(output_chunks)
        loc_payload = loc_payloads[0]
        if self._has_missing_label(loc_payload):
            return "".join(output_chunks)
        entries = self._parse_bbox_2d_entries(loc_payload)
        if not entries or self._has_invalid_box(entries):
            return "".join(output_chunks)

        current_entries = entries
        for _ in range(self.loc_verifier_rounds):
            entries_by_image: Dict[int, List[LocEntry]] = {}
            for entry in current_entries:
                entries_by_image.setdefault(entry.image_index, []).append(entry)

            original_images: List[Image.Image] = []
            overlay_images: List[Image.Image] = []
            for image_idx in range(1, len(visuals) + 1):
                original = visuals[image_idx - 1].convert("RGB")
                image_entries = entries_by_image.get(image_idx, [])
                overlay = self._overlay_boxes(original, image_entries)
                original_images.append(original)
                overlay_images.append(overlay)

            verifier_prompt = self._build_verifier_prompt(current_entries, len(visuals))
            verifier_output = self._run_verifier(original_images, overlay_images, verifier_prompt)
            corrections, _, _ = self._parse_verifier_feedback(verifier_output, current_entries)

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I have some feedback for you to incorporate. "
                                f"Please output ONLY one <bbox_2d> block using lines formatted as {bbox_line_format} "
                                "that incorporates the feedback.\n"
                                f"Feedback: {corrections}\n"
                                "You MUST re-predict ALL boxes, including unchanged ones, and keep indices sequential "
                                "starting at 1. You MUST incorporate the feedback and MUST NOT make unrelated changes."
                            ),
                        }
                    ],
                }
            )
            remaining = max_total_tokens - total_generated_tokens
            if remaining <= 0:
                return "".join(output_chunks)
            correction_chunk, correction_tokens = self._generate_once(
                messages,
                gen_kwargs,
                stop_sequences=["</bbox_2d>"],
                max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
            )
            total_generated_tokens += correction_tokens
            output_chunks.append(correction_chunk)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": correction_chunk}]})

            corrected_payloads = self._extract_bbox_2d_payloads(correction_chunk)
            if len(corrected_payloads) != 1:
                continue
            corrected_payload = corrected_payloads[0]
            if self._has_missing_label(corrected_payload):
                continue
            corrected_entries = self._parse_bbox_2d_entries(corrected_payload)
            if not corrected_entries or self._has_invalid_box(corrected_entries):
                continue
            current_entries = corrected_entries

        latest_bbox_block = self._format_bbox_block(current_entries)

        answer_prompt = self._build_clean_answer_prompt(query, latest_bbox_block)
        answer_messages = self._build_messages(answer_prompt, visuals)
        current_answer_output, _ = self._generate_once(
            answer_messages,
            gen_kwargs,
            stop_sequences=["</answer>"],
            max_new_tokens=max_new_tokens_per_chunk,
        )

        for _ in range(self.logic_verifier_rounds):
            logic_prompt = self._build_logic_self_verifier_prompt(query, current_answer_output)
            logic_messages = self._build_messages(logic_prompt, visuals)
            logic_output, _ = self._generate_once(
                logic_messages,
                gen_kwargs,
                max_new_tokens=self.logic_verifier_max_new_tokens,
            )
            logic_status, logic_feedback, logic_parse_valid = self._parse_logic_self_verifier_output(logic_output)
            if not logic_parse_valid:
                logic_status = "INVALID"
                logic_feedback = "No valid self-verifier feedback was produced. Re-emit the current answer unchanged."

            rewrite_prompt = self._build_answer_rewrite_prompt(
                query=query,
                latest_bbox_block=latest_bbox_block,
                current_answer_output=current_answer_output,
                logic_feedback=logic_feedback,
            )
            rewrite_messages = self._build_messages(rewrite_prompt, visuals)
            current_answer_output, _ = self._generate_once(
                rewrite_messages,
                gen_kwargs,
                stop_sequences=["</answer>"],
                max_new_tokens=max_new_tokens_per_chunk,
            )

        output_chunks.append(current_answer_output)
        return "".join(output_chunks)
