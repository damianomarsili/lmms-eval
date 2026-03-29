import re
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_no_verifier_vllm import STTVNoVerifierVLLM

LOGIC_STEP_EDIT_PATTERN = re.compile(r"(?i)^EDIT_STEP\s+(?P<idx>\d+)\s*:\s*(?P<body>.+)$")
REASON_BLOCK_PATTERN = re.compile(r"(?is)<reason>\s*(?P<body>.*?)\s*</reason>")
REASON_STEP_LINE_PATTERN = re.compile(r"^\s*(?P<idx>\d+)\.\s*(?P<body>.+?)\s*$")


@register_model("sttv_implicit_grounding_vllm")
class STTVImplicitGroundingVLLM(STTVNoVerifierVLLM):
    """
    Eval-time mirror of verl_implicit_grounding flow:
    query -> <reason><answer> -> logic self-verifier -> rewritten <reason><answer>
    No explicit grounding / no grounding verifier / no compaction.
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
        logic_self_verifier_prompt_path: Optional[str] = None,
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
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            no_verifier_max_new_tokens=generation_chunk_max_new_tokens,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )

        if logic_self_verifier_prompt_path is None:
            prompt_file = (
                Path(__file__).resolve().parents[4]
                / "training"
                / "prompts"
                / "logic_self_verifier_gemini_implicit_grounding_instructions.txt"
            )
        else:
            prompt_file = Path(logic_self_verifier_prompt_path)
        if not prompt_file.exists():
            raise FileNotFoundError(f"Logic self-verifier prompt file not found: {prompt_file}")
        self.logic_self_verifier_template = prompt_file.read_text(encoding="utf-8").strip()
        if not self.logic_self_verifier_template:
            raise ValueError(f"Logic self-verifier prompt file is empty: {prompt_file}")

        self.logic_verifier_rounds = max(0, int(logic_verifier_rounds))
        self.logic_verifier_max_new_tokens = int(logic_verifier_max_new_tokens)
        self.generation_chunk_max_new_tokens = int(generation_chunk_max_new_tokens)

    def _build_initial_answer_prompt(self, query: str) -> str:
        query_text = query.strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags with numbered steps "
            "(1., 2., 3., ... one step per line), then putting ONLY your final answer inside <answer>. "
            "Use the answer type required by the question: option text when options are provided, a single number "
            "for quantitative questions, and yes/no only for explicit binary questions. "
            "Do not answer unknown/impossible; provide your best estimate. "
            "For counting questions, count only clearly visible instances once; do not infer hidden or distant instances. "
            "Output exactly one non-empty and CLOSED numbered <reason>...</reason> and one <answer>...</answer>. "
            "Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )

    def _build_logic_self_verifier_prompt(self, query: str, latest_answer_output: str) -> str:
        return self.logic_self_verifier_template.format(
            query=query.strip(),
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

    def _parse_logic_step_edits_optional(self, text: str, current_answer_output: str) -> tuple[str, bool]:
        cleaned = str(text or "").replace("<|im_end|>", "").strip()
        normalized_lines: List[tuple[str, int]] = []
        line_order = 0
        saw_nonempty_line = False
        valid_step_indices = set(self._extract_reason_step_indices(current_answer_output))
        seen: set[str] = set()

        for raw_line in cleaned.splitlines():
            if len(normalized_lines) >= 2:
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
            if reason_match is None:
                continue
            try:
                step_idx = int(reason_match.group("idx"))
            except (TypeError, ValueError):
                continue
            if valid_step_indices and step_idx not in valid_step_indices:
                continue
            body = reason_match.group("body").strip()
            if not body:
                continue
            normalized = f"EDIT_STEP {step_idx}: {body}"
            if normalized in seen:
                continue
            normalized_lines.append((normalized, line_order))
            seen.add(normalized)
            line_order += 1

        normalized_lines.sort(key=lambda item: item[1])
        parse_valid = bool(normalized_lines or not saw_nonempty_line)
        if not normalized_lines:
            return "", parse_valid
        return "\n".join(line for line, _ in normalized_lines), True

    def _build_answer_rewrite_prompt(self, query: str, current_answer_output: str, logic_feedback: str) -> str:
        return (
            f"{query.strip()}\n\n"
            f"Current answer draft:\n{str(current_answer_output or '').strip()}\n\n"
            "I have some feedback for you to incorporate. "
            "Please update the <reason> using the feedback, revising the referenced numbered reasoning steps only, "
            "then output a final <answer> that follows from the updated reasoning.\n"
            f"Feedback:\n{logic_feedback}\n"
            "You MUST update the reasoning to incorporate the feedback. "
            "You MUST keep the <reason> step-indexed with numbered lines (1., 2., 3., ... one step per line). "
            "You MUST then produce the final answer from that updated reasoning. "
            "You MUST incorporate the feedback and MUST NOT make unrelated changes. "
            "Apply the feedback silently: do not mention feedback, instructions, or edits in <reason>. "
            "Keep <reason> concise and non-repetitive. "
            "Use the answer type required by the question: option text when options are provided, a single number "
            "for quantitative questions, and yes/no only for explicit binary questions. "
            "Do not answer unknown/impossible; provide your best estimate. "
            "For counting questions, count only clearly visible instances once; do not infer hidden or distant instances. "
            "Output exactly one non-empty and CLOSED numbered <reason>...</reason> and one <answer>...</answer>. "
            "Unless explicitly specified otherwise, assume all metric quantities are 3D and depth-aware. "
            "Do not round answers, express all ratios as unrounded decimals. Nothing else."
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            return -len(x[0]), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        self.last_generation_metadata = None
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

                for _ in range(self.logic_verifier_rounds):
                    logic_prompt = self._build_logic_self_verifier_prompt(context, current_answer_output)
                    logic_messages = self._build_messages(logic_prompt, visuals)
                    logic_output, _ = self._generate_once(
                        logic_messages,
                        sample_gen_kwargs,
                        max_new_tokens=max(1, self.logic_verifier_max_new_tokens),
                    )
                    logic_feedback, logic_parse_valid = self._parse_logic_step_edits_optional(
                        logic_output, current_answer_output
                    )
                    if (not logic_parse_valid) or (not str(logic_feedback or "").strip()):
                        logic_feedback = "No valid self-verifier feedback was produced. Re-emit the current answer unchanged."

                    rewrite_prompt = self._build_answer_rewrite_prompt(context, current_answer_output, logic_feedback)
                    rewrite_messages = self._build_messages(rewrite_prompt, visuals)
                    current_answer_output, _ = self._generate_once(
                        rewrite_messages,
                        sample_gen_kwargs,
                        stop_sequences=["</answer>"],
                        max_new_tokens=max(1, self.generation_chunk_max_new_tokens),
                    )

                res.append(current_answer_output)
                self.cache_hook.add_partial("generate_until", (context, sample_gen_kwargs), current_answer_output)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        pbar.close()
        return res
