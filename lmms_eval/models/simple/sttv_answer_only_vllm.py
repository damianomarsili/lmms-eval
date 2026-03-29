import re
from typing import Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.sttv_no_verifier_vllm import STTVNoVerifierVLLM


@register_model("sttv_answer_only_vllm")
class STTVAnswerOnlyVLLM(STTVNoVerifierVLLM):
    """
    STTV answer-only vLLM model:
    - one pass
    - no grounding / no verifier
    - prompt asks directly for <reason> then <answer>
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        depth: Union[bool, str, int, float] = False,
        instruction_mode: str = "box",
        max_image_side: int = 768,
        no_verifier_max_new_tokens: int = 768,
        self_consistency_k: int = 1,
        self_consistency_temperature: float = 0.7,
        self_consistency_top_p: float = 0.95,
        self_consistency_selector: str = "majority_vote",
        self_consistency_judge_max_new_tokens: int = 64,
        self_consistency_judge_temperature: float = 0.0,
        self_consistency_judge_top_p: float = 1.0,
        self_consistency_judge_max_prompt_tokens: int = 2048,
        self_consistency_judge_candidate_max_chars: int = 1200,
        self_consistency_judge_cleanup_cuda: Union[bool, str, int] = True,
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
            prompt_path=None,
            instruction_mode=instruction_mode,
            max_image_side=max_image_side,
            no_verifier_max_new_tokens=no_verifier_max_new_tokens,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
            min_image_pixels=min_image_pixels,
            seed=seed,
            disable_log_stats=disable_log_stats,
            **kwargs,
        )
        self.self_consistency_k = max(1, int(self_consistency_k))
        self.self_consistency_temperature = float(self_consistency_temperature)
        self.self_consistency_top_p = float(self_consistency_top_p)
        selector = str(self_consistency_selector or "majority_vote").strip().lower()
        selector_aliases = {
            "majority": "majority_vote",
            "majority_vote": "majority_vote",
            "llm_judge": "llm_judge",
            "judge": "llm_judge",
        }
        if selector not in selector_aliases:
            raise ValueError(
                f"Unsupported self_consistency_selector={self_consistency_selector!r}. "
                "Use one of: majority_vote, llm_judge."
            )
        self.self_consistency_selector = selector_aliases[selector]
        self.self_consistency_judge_max_new_tokens = max(8, int(self_consistency_judge_max_new_tokens))
        self.self_consistency_judge_temperature = float(self_consistency_judge_temperature)
        self.self_consistency_judge_top_p = float(self_consistency_judge_top_p)
        self.self_consistency_judge_max_prompt_tokens = max(256, int(self_consistency_judge_max_prompt_tokens))
        self.self_consistency_judge_candidate_max_chars = max(128, int(self_consistency_judge_candidate_max_chars))
        self.self_consistency_judge_cleanup_cuda = str(self_consistency_judge_cleanup_cuda).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _build_prompted_context(self, query: str) -> str:
        query_text = query.strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags and then putting ONLY your final answer "
            "inside <answer>. Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )

    def _vote_key(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        answer_blocks = re.findall(r"(?is)<answer>\s*(.*?)\s*</answer>", cleaned)
        if answer_blocks:
            cleaned = answer_blocks[-1].strip()
        return " ".join(cleaned.lower().split())

    def _majority_vote_text(self, candidates: List[str]) -> str:
        if len(candidates) == 0:
            return ""
        counts: Dict[str, int] = {}
        first_idx: Dict[str, int] = {}
        chosen_text: Dict[str, str] = {}
        for idx, text in enumerate(candidates):
            key = self._vote_key(text)
            counts[key] = counts.get(key, 0) + 1
            if key not in first_idx:
                first_idx[key] = idx
                chosen_text[key] = text
        best_key = max(counts.keys(), key=lambda k: (counts[k], -first_idx[k]))
        return chosen_text[best_key]

    def _build_llm_judge_prompt(self, query: str, candidates: List[str]) -> str:
        def _extract_tag(text: str, tag: str) -> str:
            blocks = re.findall(rf"(?is)<{tag}>\s*(.*?)\s*</{tag}>", str(text or ""))
            return blocks[-1].strip() if blocks else ""

        def _compact_candidate(text: str) -> str:
            raw = str(text or "").strip()
            answer = _extract_tag(raw, "answer")
            reason = _extract_tag(raw, "reason")
            if not answer:
                answer = raw
            answer = answer[: self.self_consistency_judge_candidate_max_chars].strip()
            if reason:
                reason = reason[: max(128, self.self_consistency_judge_candidate_max_chars // 2)].strip()
            if reason:
                return f"<reason>{reason}</reason>\n<answer>{answer}</answer>"
            return f"<answer>{answer}</answer>"

        lines = [
            "You are an answer judge.",
            "Given a query and candidate answers, pick the single best candidate.",
            "Focus on correctness and adherence to the requested answer format.",
            "",
            "Query:",
            str(query or "").strip(),
            "",
            "Candidates:",
        ]
        for idx, text in enumerate(candidates, start=1):
            lines.extend([f"[{idx}]", _compact_candidate(text), ""])
        lines.append(f"Return exactly one line in this format: BEST: <index 1-{len(candidates)}>")
        prompt = "\n".join(lines)
        max_chars = int(self.self_consistency_judge_max_prompt_tokens * 4)
        if len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
        return prompt

    def _parse_llm_judge_choice(self, judge_text: str, num_candidates: int) -> Optional[int]:
        if num_candidates <= 0:
            return None
        text = str(judge_text or "")
        match = re.search(r"(?i)\bBEST\b\s*[:#-]?\s*(\d+)", text)
        if match is None:
            match = re.search(r"\b(\d+)\b", text)
        if match is None:
            return None
        try:
            one_based = int(match.group(1))
        except (TypeError, ValueError):
            return None
        if one_based < 1 or one_based > num_candidates:
            return None
        return one_based - 1

    def _override_sampling(self, *, greedy: bool, temperature: float, top_p: float) -> Tuple[bool, float, float]:
        prev = (self.generation_greedy, self.generation_temperature, self.generation_top_p)
        self.generation_greedy = bool(greedy)
        self.generation_temperature = float(temperature)
        self.generation_top_p = float(top_p)
        return prev

    def _restore_sampling(self, prev: Tuple[bool, float, float]) -> None:
        self.generation_greedy, self.generation_temperature, self.generation_top_p = prev

    def _select_with_llm_judge_batched(
        self,
        queries: List[str],
        candidates_by_sample: List[List[str]],
    ) -> List[str]:
        judge_messages: List[List[Dict[str, object]]] = []
        valid_indices: List[int] = []
        selected: List[str] = [self._majority_vote_text(cands) for cands in candidates_by_sample]

        for idx, (query, cands) in enumerate(zip(queries, candidates_by_sample)):
            if len(cands) <= 1:
                selected[idx] = cands[0] if cands else ""
                continue
            prompt = self._build_llm_judge_prompt(query, cands)
            judge_messages.append(self._build_messages(prompt, None))
            valid_indices.append(idx)

        if not judge_messages:
            return selected

        judge_greedy = self.self_consistency_judge_temperature <= 0.0
        prev_sampling = self._override_sampling(
            greedy=judge_greedy,
            temperature=self.self_consistency_judge_temperature,
            top_p=self.self_consistency_judge_top_p if self.self_consistency_judge_top_p is not None else 1.0,
        )
        try:
            judge_outputs = self._generate_batch_once(
                judge_messages,
                {},
                max_new_tokens=self.self_consistency_judge_max_new_tokens,
            )
        finally:
            self._restore_sampling(prev_sampling)

        for local_idx, (judge_text, _) in enumerate(judge_outputs):
            sample_idx = valid_indices[local_idx]
            cands = candidates_by_sample[sample_idx]
            picked = self._parse_llm_judge_choice(judge_text, len(cands))
            if picked is not None:
                selected[sample_idx] = cands[picked]
            else:
                selected[sample_idx] = self._majority_vote_text(cands)
        return selected

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []

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

            batched_messages: List[List[Dict[str, object]]] = []
            cleaned_contexts: List[str] = []
            cleaned_gen_kwargs: List[Dict[str, object]] = []
            for i, context in enumerate(contexts):
                sample_task = task[i]
                sample_split = split[i]
                sample_doc_id = doc_id[i]
                sample_doc_to_visual = doc_to_visual[i]
                sample_visual = sample_doc_to_visual(self.task_dict[sample_task][sample_split][sample_doc_id])
                if "<image" in context:
                    context = re.sub(r"<image\s*\d+>", "", context)
                    context = context.replace("<image>", "")
                prompted_context = self._build_prompted_context(context)
                batched_messages.append(self._build_messages(prompted_context, sample_visual))
                cleaned_contexts.append(context)
                cleaned_gen_kwargs.append(dict(all_gen_kwargs[i]))

            all_gen_kwargs_equal = all(kwargs == cleaned_gen_kwargs[0] for kwargs in cleaned_gen_kwargs)
            if not all_gen_kwargs_equal:
                for i in range(len(batched_messages)):
                    single_outputs = self._generate_batch_once(
                        [batched_messages[i]],
                        cleaned_gen_kwargs[i],
                        stop_sequences=["</answer>"],
                        max_new_tokens=int(cleaned_gen_kwargs[i].get("max_new_tokens", self.no_verifier_max_new_tokens)),
                    )
                    answer = single_outputs[0][0] if single_outputs else ""
                    res.append(answer)
                    self.cache_hook.add_partial("generate_until", (cleaned_contexts[i], cleaned_gen_kwargs[i]), answer)
                    pbar.update(1)
                    self._cleanup_after_sample()
                continue

            base_kwargs = cleaned_gen_kwargs[0]
            requested_max_new_tokens = int(base_kwargs.get("max_new_tokens", self.no_verifier_max_new_tokens))
            try:
                base_temperature = float(base_kwargs.get("temperature", 0.0))
            except (TypeError, ValueError):
                base_temperature = 0.0
            top_p_raw = base_kwargs.get("top_p", None)
            try:
                base_top_p = float(top_p_raw) if top_p_raw is not None else 1.0
            except (TypeError, ValueError):
                base_top_p = 1.0

            if self.self_consistency_k > 1:
                sample_temperature = base_temperature if base_temperature > 0.0 else self.self_consistency_temperature
                sample_top_p = base_top_p if top_p_raw is not None else self.self_consistency_top_p
                sample_greedy = False
            else:
                sample_temperature = base_temperature
                sample_top_p = base_top_p
                sample_greedy = base_temperature <= 0.0
                if sample_greedy:
                    sample_top_p = 1.0

            candidates_by_sample: List[List[str]] = [[] for _ in range(len(batched_messages))]
            prev_sampling = self._override_sampling(
                greedy=sample_greedy,
                temperature=sample_temperature,
                top_p=sample_top_p,
            )
            try:
                for _ in range(self.self_consistency_k):
                    sampled_outputs = self._generate_batch_once(
                        batched_messages,
                        base_kwargs,
                        stop_sequences=["</answer>"],
                        max_new_tokens=requested_max_new_tokens,
                    )
                    for idx, (text, _) in enumerate(sampled_outputs):
                        candidates_by_sample[idx].append(text)
            finally:
                self._restore_sampling(prev_sampling)

            if self.self_consistency_selector == "llm_judge" and self.self_consistency_k > 1:
                selected_answers = self._select_with_llm_judge_batched(cleaned_contexts, candidates_by_sample)
            else:
                selected_answers = [self._majority_vote_text(cands) for cands in candidates_by_sample]

            for i, answer in enumerate(selected_answers):
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (cleaned_contexts[i], cleaned_gen_kwargs[i]), answer)
                pbar.update(1)
                self._cleanup_after_sample()

        res = re_ords.get_original(res)
        pbar.close()
        return res
