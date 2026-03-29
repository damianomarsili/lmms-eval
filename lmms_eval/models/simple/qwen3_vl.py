import gc
import re
from typing import Dict, List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


@register_model("qwen3_vl")
class Qwen3VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_image_side: int = 768,
        trust_remote_code: Optional[bool] = True,
        answer_only_format: Union[bool, str, int] = False,
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
        **kwargs,
    ) -> None:
        super().__init__()
        if "answer_only_format" in kwargs:
            answer_only_format = kwargs.pop("answer_only_format")
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self.device_map = device_map

        self._model = AutoModelForVision2Seq.from_pretrained(
            pretrained,
            device_map=self.device_map,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer = self.processor.tokenizer
        if self._tokenizer is None:
            raise ValueError("AutoProcessor did not provide a tokenizer.")

        self.pretrained = pretrained
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.max_image_side = max_image_side
        self.answer_only_format = str(answer_only_format).strip().lower() in {"1", "true", "yes", "on"}
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
        self.model_context_length = self._resolve_model_context_length()

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        self._device = self.model.device

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3VL")

    def _resize_longest_side(self, image: Image.Image, longest_side: int) -> Image.Image:
        width, height = image.size
        if max(width, height) <= longest_side:
            return image
        if width >= height:
            new_width = longest_side
            new_height = int(round(height * (longest_side / width)))
        else:
            new_height = longest_side
            new_width = int(round(width * (longest_side / height)))
        return image.resize((new_width, new_height), Image.BICUBIC)

    def _build_messages(self, context: str, visual: Optional[Union[List[object], object]]) -> List[Dict[str, object]]:
        content_parts: List[Dict[str, object]] = []
        if visual is not None:
            visual_items = visual if isinstance(visual, (list, tuple)) else [visual]
            for item in visual_items:
                if isinstance(item, Image.Image):
                    resized = self._resize_longest_side(item.convert("RGB"), self.max_image_side)
                    content_parts.append({"type": "image", "image": resized})
                else:
                    eval_logger.warning(f"Unsupported visual type: {type(item)}")
        content_parts.append({"type": "text", "text": context})
        return [{"role": "user", "content": content_parts}]

    def _generate_once(
        self,
        messages: List[Dict[str, object]],
        gen_kwargs: Dict[str, object],
        max_prompt_tokens: Optional[int] = None,
        cleanup_cuda: bool = False,
    ) -> str:
        if cleanup_cuda and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Enforce prompt length bounds for text-only judge calls to avoid OOM.
        is_multimodal = any(
            key in inputs for key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")
        )
        if not is_multimodal and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask", None)
            prompt_len = int(input_ids.shape[-1])
            if max_prompt_tokens is not None and prompt_len > int(max_prompt_tokens):
                keep = int(max_prompt_tokens)
                input_ids = input_ids[:, -keep:]
                if attn is not None:
                    attn = attn[:, -keep:]
                inputs["input_ids"] = input_ids
                if attn is not None:
                    inputs["attention_mask"] = attn
                prompt_len = keep

            max_new_tokens = int(gen_kwargs.get("max_new_tokens", 512))
            if self.model_context_length is not None and (prompt_len + max_new_tokens) > self.model_context_length:
                keep = max(1, int(self.model_context_length - max_new_tokens))
                if keep < prompt_len:
                    input_ids = inputs["input_ids"][:, -keep:]
                    attn = inputs.get("attention_mask", None)
                    if attn is not None:
                        attn = attn[:, -keep:]
                    inputs["input_ids"] = input_ids
                    if attn is not None:
                        inputs["attention_mask"] = attn

        inputs = inputs.to(self.model.device)

        max_new_tokens = int(gen_kwargs.get("max_new_tokens", 512))
        temperature = float(gen_kwargs.get("temperature", 0.0))
        top_p = gen_kwargs.get("top_p", None)
        top_p = float(top_p) if top_p is not None else None

        do_sample = temperature > 0.0
        generation_kwargs: Dict[str, object] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p

        cont = self.model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if cleanup_cuda and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        return answers[0]

    def _resolve_model_context_length(self) -> Optional[int]:
        candidates: List[int] = []
        for cfg in (self._config, getattr(self._config, "text_config", None)):
            if cfg is None:
                continue
            for attr in ("max_position_embeddings", "max_sequence_length", "seq_length", "model_max_length"):
                value = getattr(cfg, attr, None)
                if value is None:
                    continue
                try:
                    val = int(value)
                except (TypeError, ValueError):
                    continue
                if 0 < val < 10_000_000:
                    candidates.append(val)
        tokenizer_max_len = getattr(self._tokenizer, "model_max_length", None)
        try:
            tokenizer_max_len_int = int(tokenizer_max_len) if tokenizer_max_len is not None else None
        except (TypeError, ValueError):
            tokenizer_max_len_int = None
        if tokenizer_max_len_int is not None and 0 < tokenizer_max_len_int < 10_000_000:
            candidates.append(tokenizer_max_len_int)
        if not candidates:
            return None
        return min(candidates)

    def _vote_key(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        answer_blocks = re.findall(r"(?is)<answer>\s*(.*?)\s*</answer>", cleaned)
        if answer_blocks:
            cleaned = answer_blocks[-1].strip()
        return " ".join(cleaned.lower().split())

    def _build_answer_only_context(self, query: str) -> str:
        query_text = str(query or "").strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags and then putting ONLY your final answer "
            "inside <answer>. Ensure that the answer is either yes/no, one word, or one number. "
            "Do not round answers, express all ratios as unrounded decimals. "
            "Nothing else."
        )

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
        return "\n".join(lines)

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

    def _llm_judge_select_text(self, query: str, candidates: List[str]) -> str:
        if len(candidates) == 0:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        judge_prompt = self._build_llm_judge_prompt(query, candidates)
        judge_messages = self._build_messages(judge_prompt, None)
        judge_kwargs: Dict[str, object] = {
            "max_new_tokens": self.self_consistency_judge_max_new_tokens,
            "temperature": self.self_consistency_judge_temperature,
        }
        if self.self_consistency_judge_top_p is not None:
            judge_kwargs["top_p"] = self.self_consistency_judge_top_p
        judge_text = self._generate_once(
            judge_messages,
            judge_kwargs,
            max_prompt_tokens=self.self_consistency_judge_max_prompt_tokens,
            cleanup_cuda=self.self_consistency_judge_cleanup_cuda,
        )
        selected_idx = self._parse_llm_judge_choice(judge_text, len(candidates))
        if selected_idx is None:
            return self._majority_vote_text(candidates)
        return candidates[selected_idx]

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
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if isinstance(context, str) and "<image" in context:
                    context = re.sub(r"<image\s*\d*>", "", context).replace("<image>", "")
                prompted_context = self._build_answer_only_context(context) if self.answer_only_format else context
                messages = self._build_messages(prompted_context, visual_list[i])
                sample_kwargs = dict(gen_kwargs)
                if self.self_consistency_k > 1:
                    temp_value = sample_kwargs.get("temperature", None)
                    try:
                        temp_numeric = float(temp_value) if temp_value is not None else 0.0
                    except (TypeError, ValueError):
                        temp_numeric = 0.0
                    if temp_numeric <= 0.0:
                        sample_kwargs["temperature"] = self.self_consistency_temperature
                    if sample_kwargs.get("top_p") is None:
                        sample_kwargs["top_p"] = self.self_consistency_top_p
                candidates = [
                    self._generate_once(messages, dict(sample_kwargs))
                    for _ in range(self.self_consistency_k)
                ]
                if self.self_consistency_selector == "llm_judge" and self.self_consistency_k > 1:
                    answer = self._llm_judge_select_text(prompted_context, candidates)
                else:
                    answer = self._majority_vote_text(candidates)
                res.append(answer)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), answer)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
