import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_vllm import BBOX_2D_ENTRY_PATTERN, LocEntry, STTVVLLM

LOGIC_STEP_EDIT_PATTERN = re.compile(r"(?i)^EDIT_STEP\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
REASON_BLOCK_PATTERN = re.compile(r"(?is)<reason>\s*(?P<body>.*?)\s*</reason>")
REASON_STEP_LINE_PATTERN = re.compile(r"^\s*(?P<idx>\d+)\.\s*(?P<body>.+?)\s*$")


@register_model("sttv_all_verifiers_vllm")
class STTVAllVerifiersVLLM(STTVVLLM):
    _ROBOSPATIAL_COMPATIBILITY_POST_PROMPT = (
        "Assume objects can be moved and re-placed. "
        "Judge whether the target placement is physically feasible, not whether objects are currently in that relation. "
        "Use bounding boxes mainly for size/clearance checks and use the image itself for semantic/depth/occlusion judgment."
    )
    _ROBOSPATIAL_CONFIGURATION_POST_PROMPT = (
        "Bounding boxes are 2D only, x is left/right, y is above/below. "
        "Greater x values are more right, greater y values are lower. "
        "Do not use bounding box values x or y to infer depth relations like behind/in-front. "
        "Use the image for this.\n"
        "Do not over-rely on the boxes for semantic relations, use the boxes to know where to look at the image "
        "and evaluate semantic relations based on the image itself not just the boxes!"
    )
    _BLINK_COUNTING_POST_PROMPT = (
        "Do not over-rely on the boxes for semantic relations, use the boxes to know where to look at the image "
        "and evaluate semantic relations based on the image itself not just the boxes!"
    )
    _BLINK_SPATIAL_POST_PROMPT = (
        "Use the boxes as a guide to locate objects, then decide the asked relation from image semantics. "
        "Do not infer depth or orientation solely from x/y box coordinates."
    )
    _OMNI3D_POST_PROMPT = (
        "When options are provided, your final answer must be exactly one option token from the provided set (no paraphrase, no extra words). "
        "For questions that ask \"How many\", your final answer must be a number. "
        "For any numeric question, output exactly one integer or decimal number only (no fractions like a/b, no units, no extra text). "
        "If exact counting is uncertain, provide your best numeric estimate instead of refusing. "
        "Use the provided bounding boxes as a guide, but count from the image itself because detections may be incomplete."
    )
    _REALWORLDQA_POST_PROMPT = (
        "After detecting objects, you MUST think step by step in a non-empty <reason>...</reason> block with numbered steps "
        "(1., 2., 3., ...), and then output your final answer in <answer>...</answer>. "
        "Do not skip the <reason> block."
    )
    _VSR_POST_PROMPT = (
        "For this binary relation check, output 1 only when the relation is clearly supported by visual evidence and no strong contradictory cue exists; "
        "otherwise output 0. Use image semantics (occlusion, relative depth, object orientation/axis, boundary contact) and do not rely on rough box "
        "overlap or x/y ordering alone. "
        "For this true/false relation check, output 1 only if the relation is clearly and unambiguously visible in the image; "
        "if evidence is partial, occluded, ambiguous, or based only on rough box overlap/order, output 0. "
        "After detecting objects, you MUST provide a non-empty step-by-step <reason>...</reason> with numbered steps, "
        "then output only 0 or 1 in <answer>...</answer>."
    )
    _VSR_FRONT_BACK_POST_PROMPT = (
        "For in-front-of/behind relations, use depth cues only: occlusion (who blocks whom), relative scale for similar object types, "
        "and perspective context. Do not decide front/behind from 2D x/y ordering or overlap alone. Output 1 only when at least one "
        "strong depth cue supports the relation and no strong cue contradicts it; otherwise output 0."
    )
    _CV_BENCH_POST_PROMPT = (
        "Bounding boxes are a starting guide only. Do not use x/y coordinates alone to infer depth ordering (closer/farther, front/behind); "
        "use visual depth cues from the image (occlusion, relative scale for similar objects, perspective). "
        "For counting questions, count from the image itself and use boxes only as guidance, since detections can be noisy or incomplete. "
        "Count unique instances only: do not count duplicate/overlapping boxes of the same object as multiple instances, and do not count partial fragments as separate objects. "
        "For relation questions, evaluate the first named object/entity with respect to the second exactly as asked, and do not invert subject/object roles. "
        "Before outputting the final option, run one consistency check that the option text matches your computed count/relation, then output only that option."
    )
    _OMNISPATIAL_POST_PROMPT = (
        "You MUST always choose exactly one provided option (A/B/C/D), even when uncertain; do not refuse and do not output "
        "\"cannot determine\" unless that exact option is the best choice among the provided options. "
        "After detecting objects, you MUST provide a non-empty step-by-step <reason>...</reason> with numbered steps "
        "(1., 2., 3., ...), then output exactly one option letter inside <answer>...</answer>."
    )
    _COUNTBENCHQA_LOGIC_POST_PROMPT = (
        "For counting questions, explicitly check whether multiple boxes of the same label refer to the same object. "
        "If boxes are duplicate detections of one object (heavy overlap / near-identical placement), refine the reasoning "
        "to count unique objects only and avoid overcounting."
    )

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
        logic_verifier_max_step_edits: int = 2,
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
        self.logic_verifier_max_step_edits = max(0, int(logic_verifier_max_step_edits))
        self.logic_self_verifier_template = self._load_logic_self_verifier_template(logic_self_verifier_prompt_path)

    def _load_logic_self_verifier_template(self, prompt_path: Optional[str]) -> str:
        if prompt_path is None:
            repo_root = Path(__file__).resolve().parents[4]
            prompt_file = repo_root / "training" / "prompts" / "logic_self_verifier_gemini_instructions.txt"
        else:
            prompt_file = Path(prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Logic self-verifier prompt file not found: {prompt_file}")
        prompt_text = prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Logic self-verifier prompt file is empty: {prompt_file}")
        return prompt_text

    def _build_clean_answer_prompt(self, query: str, latest_bbox_block: str, task_name: str) -> str:
        query_text = self._augment_query_with_task_post_prompt(query, task_name).strip()
        return (
            f"Original query:\n{query_text}\n\n"
            "Detected objects (in [x_min, y_min, x_max, y_max] format with coordinates in [0,1000]):\n"
            f"{latest_bbox_block}\n\n"
            f"Here is the query again:\n{query_text}\n\n"
            "Please now answer the query by first reasoning inside <reason> tags using numbered steps "
            "(1., 2., 3., ... one step per line) and then putting ONLY your final answer inside <answer>. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Do not output another <bbox_2d>."
        )

    def _build_logic_self_verifier_prompt(
        self, query: str, latest_bbox_block: str, latest_answer_output: str, task_name: str
    ) -> str:
        query_text = self._augment_query_with_task_post_prompt(query, task_name).strip()
        if str(task_name or "").strip().lower() == "countbenchqa":
            query_text = f"{query_text}\n\n{self._COUNTBENCHQA_LOGIC_POST_PROMPT}"
        return self.logic_self_verifier_template.format(
            query=query_text,
            detected_objects=str(latest_bbox_block or "").strip(),
            answer=str(latest_answer_output or "").strip(),
        )

    def _extract_reason_step_indices(self, answer_output: str) -> List[int]:
        cleaned = str(answer_output or "").replace("<|im_end|>", "").strip()
        match = REASON_BLOCK_PATTERN.search(cleaned)
        if match is None:
            return []
        step_indices: List[int] = []
        for raw_line in match.group("body").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            step_match = REASON_STEP_LINE_PATTERN.fullmatch(line)
            if step_match is None:
                continue
            try:
                step_idx = int(step_match.group("idx"))
            except (TypeError, ValueError):
                continue
            if step_idx not in step_indices:
                step_indices.append(step_idx)
        return step_indices

    def _parse_logic_self_verifier_output(
        self, text: str, current_answer_output: str
    ) -> Tuple[str, bool]:
        cleaned = str(text or "").replace("<|im_end|>", "").strip()
        normalized_lines: List[Tuple[str, int]] = []
        line_order = 0
        seen: set[str] = set()
        saw_nonempty_line = False
        valid_step_indices = set(self._extract_reason_step_indices(current_answer_output))

        for raw_line in cleaned.splitlines():
            if len(normalized_lines) >= self.logic_verifier_max_step_edits:
                break
            line = raw_line.strip()
            if not line:
                continue
            saw_nonempty_line = True
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not line:
                continue
            if line.lower().startswith("feedback:"):
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue

            reason_match = LOGIC_STEP_EDIT_PATTERN.match(line)
            if reason_match is not None:
                try:
                    step_idx = int(reason_match.group("idx"))
                except (TypeError, ValueError):
                    continue
                if valid_step_indices and step_idx not in valid_step_indices:
                    continue
                body = reason_match.group("body").strip()
                if body:
                    normalized = f"EDIT_STEP {step_idx}: {body}"
                    if normalized not in seen:
                        normalized_lines.append((normalized, line_order))
                        seen.add(normalized)
                        line_order += 1
                continue

        normalized_lines.sort(key=lambda item: item[1])
        parse_valid = bool(normalized_lines or not saw_nonempty_line)
        if not normalized_lines:
            return "", parse_valid
        return "\n".join(line for line, _ in normalized_lines), True

    def _build_answer_rewrite_prompt(
        self,
        query: str,
        latest_bbox_block: str,
        current_answer_output: str,
        logic_feedback: str,
        task_name: str,
    ) -> str:
        clean_prompt = self._build_clean_answer_prompt(query, latest_bbox_block, task_name)
        return (
            f"{clean_prompt}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            "I have some feedback for you to incorporate. "
            "Please update the <reason> using the feedback, revising the referenced numbered reasoning steps only, "
            "then output a final <answer> that follows from the updated reasoning.\n"
            f"Feedback:\n{logic_feedback}\n"
            "You MUST update the reasoning to incorporate the feedback. "
            "If the feedback is empty, non-concrete, or does not identify a checkable error, re-emit the current answer draft unchanged. "
            "You MUST keep the <reason> step-indexed with numbered lines (1., 2., 3., ... one step per line). "
            "You MUST then produce the final answer from that updated reasoning. "
            "Do NOT change the final <answer> unless the corrected reasoning changes the computed result. "
            "You MUST incorporate the feedback and MUST NOT make unrelated changes. "
            "Please output exactly one full <reason> block and then one full <answer> block. "
            "Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else. "
            "Do not output any <bbox_2d>."
        )

    def _extract_partial_bbox_2d_payload(self, text: str) -> str:
        raw_text = str(text or "")
        lowered = raw_text.lower()
        open_tag = "<bbox_2d>"
        close_tag = "</bbox_2d>"
        start = lowered.find(open_tag)
        if start == -1:
            return ""
        start += len(open_tag)
        end = lowered.find(close_tag, start)
        if end == -1:
            end = len(raw_text)
        return raw_text[start:end].strip()

    def _parse_bbox_2d_entries_relaxed(self, payload: str) -> List[LocEntry]:
        entries: List[LocEntry] = []
        for raw_line in str(payload or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = BBOX_2D_ENTRY_PATTERN.fullmatch(line)
            if match is None:
                if entries:
                    break
                return []
            try:
                idx = int(match.group("idx"))
            except (TypeError, ValueError):
                if entries:
                    break
                return []
            if idx != len(entries) + 1:
                if entries:
                    break
                return []

            label = match.group("label").strip()
            if not label:
                if entries:
                    break
                return []
            numbers = re.findall(r"-?\d+(?:\.\d+)?", match.group("coords"))
            if len(numbers) != 4:
                if entries:
                    break
                return []
            coords_tuple = tuple(float(n) for n in numbers[:4])
            x1, y1, x2, y2 = coords_tuple
            if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
                if entries:
                    break
                return []
            if x2 <= x1 or y2 <= y1:
                if entries:
                    break
                return []
            entries.append(LocEntry(image_index=1, label=label, coords=coords_tuple))
        return entries

    def _get_task_specific_post_prompt(self, task_name: str) -> str:
        task_key = str(task_name or "").strip().lower()
        if task_key == "robospatial_compatibility":
            return self._ROBOSPATIAL_COMPATIBILITY_POST_PROMPT
        if task_key == "robospatial_configuration":
            return self._ROBOSPATIAL_CONFIGURATION_POST_PROMPT
        if task_key == "blink_counting":
            return self._BLINK_COUNTING_POST_PROMPT
        if task_key == "blink_spatial_relation":
            return self._BLINK_SPATIAL_POST_PROMPT
        if task_key == "omni3d_bench":
            return self._OMNI3D_POST_PROMPT
        if task_key == "realworldqa":
            return self._REALWORLDQA_POST_PROMPT
        if task_key == "vsr":
            return self._VSR_POST_PROMPT
        if task_key == "omnispatial_test":
            return self._OMNISPATIAL_POST_PROMPT
        if task_key == "cv-bench":
            return self._CV_BENCH_POST_PROMPT
        return ""

    def _augment_query_with_task_post_prompt(self, query: str, task_name: str) -> str:
        base_query = str(query or "").strip()
        task_key = str(task_name or "").strip().lower()
        extras: List[str] = []
        task_prompt = self._get_task_specific_post_prompt(task_name)
        if task_prompt:
            extras.append(task_prompt)

        if task_key == "vsr":
            lowered_query = base_query.lower()
            if "in front of" in lowered_query or "behind" in lowered_query:
                extras.append(self._VSR_FRONT_BACK_POST_PROMPT)

        if not extras:
            return base_query
        return f"{base_query}\n\n" + "\n".join(extras)

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(x[0]), x[0]

        pbar = self._make_eval_progress_bar(len(requests))
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        self.last_generation_metadata = None
        all_generation_metadata: List[Dict[str, object]] = []
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task_name = task[0]
            split_name = split[0]
            grounding_only = self._is_grounding_only_task(task_name)
            visual_list = [doc_to_visual[0](self.task_dict[task_name][split_name][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if "<image" in context:
                    context = re.sub(r"<image\s*\d+>", "", context)
                    context = context.replace("<image>", "")
                query_text = self._augment_query_with_task_post_prompt(context, task_name)
                prompted_context = self._build_prompted_context(query_text)
                messages = self._build_messages(prompted_context, visual_list[i])

                visuals = visual_list[i] if isinstance(visual_list[i], list) else [visual_list[i]]
                visuals = [item for item in visuals if isinstance(item, Image.Image)]
                answer, metadata = self._generate_with_verifier(
                    messages,
                    dict(gen_kwargs),
                    visuals,
                    context,
                    task_name,
                    grounding_only=grounding_only,
                )
                res.append(answer)
                all_generation_metadata.append(metadata)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                self._advance_eval_progress(pbar)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        self.last_generation_metadata = re_ords.get_original(all_generation_metadata)
        pbar.close()
        return res

    def _generate_with_verifier(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        visuals: List[Image.Image],
        query: str,
        task_name: str,
        grounding_only: bool = False,
    ) -> Tuple[str, Dict[str, object]]:
        output_chunks: List[str] = []
        generation_metadata: Dict[str, object] = {
            "grounding_only": bool(grounding_only),
            "requested_grounding_verifier_rounds": int(self.loc_verifier_rounds),
            "performed_grounding_verifier_rounds": 0,
            "grounding_rounds": [],
        }
        max_new_tokens_per_chunk = self.generation_chunk_max_new_tokens
        bbox_line_format = '1: label="object_name", [x_min, y_min, x_max, y_max]'
        total_generated_tokens = 0
        max_total_tokens = self.generation_max_new_tokens
        grounding_rounds = generation_metadata["grounding_rounds"]
        assert isinstance(grounding_rounds, list)
        current_entries: List[LocEntry] = []

        if max_total_tokens <= 0:
            generation_metadata["grounding_stop_reason"] = "generation_budget_exhausted_before_start"
        else:
            remaining = max_total_tokens - total_generated_tokens
            chunk, new_tokens = self._generate_once(
                messages,
                gen_kwargs,
                stop_sequences=["</bbox_2d>"],
                max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
            )
            total_generated_tokens += new_tokens

            if not chunk.strip():
                generation_metadata["grounding_stop_reason"] = "empty_initial_grounding_output"
            else:
                output_chunks.append(chunk)
                messages.append({"role": "assistant", "content": [{"type": "text", "text": chunk}]})

                initial_entries: List[LocEntry] = []
                initial_parse_issue: Optional[str] = None
                initial_accepted = False

                loc_payloads = self._extract_bbox_2d_payloads(chunk)
                if len(loc_payloads) == 1:
                    loc_payload = loc_payloads[0]
                    if self._has_missing_label(loc_payload):
                        initial_parse_issue = "initial_grounding_output_missing_label"
                    else:
                        strict_entries = self._parse_bbox_2d_entries(loc_payload)
                        if strict_entries and not self._has_invalid_box(strict_entries):
                            initial_entries = strict_entries
                            initial_accepted = True
                        else:
                            initial_parse_issue = "initial_grounding_output_invalid_boxes"
                else:
                    initial_parse_issue = "initial_grounding_output_missing_single_bbox_block"

                if not initial_entries:
                    partial_payload = self._extract_partial_bbox_2d_payload(chunk)
                    recovered_entries = self._parse_bbox_2d_entries_relaxed(partial_payload)
                    if recovered_entries and not self._has_invalid_box(recovered_entries):
                        initial_entries = recovered_entries
                        generation_metadata["initial_grounding_recovered_from_partial"] = True

                current_entries = initial_entries
                grounding_rounds.append(
                    self._build_grounding_round_record(
                        stage="initial_prediction",
                        round_index=0,
                        raw_output=chunk,
                        candidate_entries=initial_entries,
                        accepted_entries=initial_entries,
                        accepted=initial_accepted,
                    )
                )
                if initial_parse_issue is not None:
                    generation_metadata["initial_grounding_issue"] = initial_parse_issue

        can_run_grounding_verifier = bool(output_chunks)
        for round_index in range(1, self.loc_verifier_rounds + 1):
            if not can_run_grounding_verifier:
                break
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
            corrections, raw_verifier_feedback, feedback_info = self._parse_verifier_feedback(verifier_output, current_entries)

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
                generation_metadata["grounding_stop_reason"] = "generation_budget_exhausted_during_grounding_verification"
                break
            correction_chunk, correction_tokens = self._generate_once(
                messages,
                gen_kwargs,
                stop_sequences=["</bbox_2d>"],
                max_new_tokens=max(1, min(max_new_tokens_per_chunk, remaining)),
            )
            total_generated_tokens += correction_tokens
            output_chunks.append(correction_chunk)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": correction_chunk}]})

            corrected_entries: List[LocEntry] = []
            accepted = False
            corrected_payloads = self._extract_bbox_2d_payloads(correction_chunk)
            if len(corrected_payloads) != 1:
                accepted_entries = current_entries
            else:
                corrected_payload = corrected_payloads[0]
                if not self._has_missing_label(corrected_payload):
                    corrected_entries = self._parse_bbox_2d_entries(corrected_payload)
                    if corrected_entries and not self._has_invalid_box(corrected_entries):
                        accepted = True
                        current_entries = corrected_entries
                accepted_entries = current_entries

            grounding_rounds.append(
                self._build_grounding_round_record(
                    stage="loc_verifier_round",
                    round_index=round_index,
                    raw_output=correction_chunk,
                    candidate_entries=corrected_entries,
                    accepted_entries=accepted_entries,
                    accepted=accepted,
                    verifier_prompt=verifier_prompt,
                    verifier_output=verifier_output,
                    parsed_corrections=corrections,
                    raw_verifier_feedback=raw_verifier_feedback,
                    feedback_info=feedback_info,
                )
            )
            generation_metadata["performed_grounding_verifier_rounds"] = round_index

        latest_bbox_block = self._format_bbox_block(current_entries)
        generation_metadata["final_bbox_block"] = latest_bbox_block
        generation_metadata["final_bbox_entries"] = self._serialize_loc_entries(current_entries)
        if grounding_only:
            generation_metadata["grounding_stop_reason"] = "grounding_only_task"
            return "".join(output_chunks), generation_metadata

        answer_prompt = self._build_clean_answer_prompt(query, latest_bbox_block, task_name)
        answer_messages = self._build_messages(answer_prompt, visuals)
        current_answer_output, _ = self._generate_once(
            answer_messages,
            gen_kwargs,
            stop_sequences=["</answer>"],
            max_new_tokens=max(1, max_new_tokens_per_chunk),
        )

        for _ in range(self.logic_verifier_rounds):
            logic_prompt = self._build_logic_self_verifier_prompt(
                query, latest_bbox_block, current_answer_output, task_name
            )
            logic_messages = self._build_messages(logic_prompt, visuals)
            logic_output, _ = self._generate_once(
                logic_messages,
                gen_kwargs,
                max_new_tokens=max(1, self.logic_verifier_max_new_tokens),
            )
            logic_feedback, logic_parse_valid = self._parse_logic_self_verifier_output(
                logic_output, current_answer_output
            )
            if (not logic_parse_valid) or (not str(logic_feedback or "").strip()):
                logic_feedback = "No valid self-verifier feedback was produced. Re-emit the current answer unchanged."

            rewrite_prompt = self._build_answer_rewrite_prompt(
                query=query,
                latest_bbox_block=latest_bbox_block,
                current_answer_output=current_answer_output,
                logic_feedback=logic_feedback,
                task_name=task_name,
            )
            rewrite_messages = self._build_messages(rewrite_prompt, visuals)
            current_answer_output, _ = self._generate_once(
                rewrite_messages,
                gen_kwargs,
                stop_sequences=["</answer>"],
                max_new_tokens=max(1, max_new_tokens_per_chunk),
            )

        output_chunks.append(current_answer_output)
        prior_stop_reason = generation_metadata.get("grounding_stop_reason")
        if isinstance(prior_stop_reason, str) and prior_stop_reason and prior_stop_reason != "continued_to_answer_generation":
            generation_metadata["grounding_pre_answer_stop_reason"] = prior_stop_reason
        generation_metadata["grounding_stop_reason"] = "continued_to_answer_generation"
        return "".join(output_chunks), generation_metadata
