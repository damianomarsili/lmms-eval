import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from loguru import logger as eval_logger
from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.openai_compatible import OpenAICompatible
from tqdm import tqdm


@register_model("sttv_answer_only_openai")
class STTVAnswerOnlyOpenAI(OpenAICompatible):
    """
    OpenAI-backed answer-only model:
    - one pass
    - no grounding / no verifier
    - asks directly for <reason> then <answer>
    """

    def __init__(
        self,
        model_version: str = "gpt-5-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        max_frames_num: int = 10,
        httpx_trust_env: bool = True,
        batch_size: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(
            model_version=model_version,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            max_size_in_mb=max_size_in_mb,
            max_frames_num=max_frames_num,
            httpx_trust_env=httpx_trust_env,
            batch_size=batch_size,
            **kwargs,
        )

    def _build_prompted_context(self, query: str) -> str:
        query_text = str(query or "").strip()
        return (
            f"{query_text}\n\n"
            "Please answer the query by first reasoning inside <reason> tags and then putting ONLY your final answer "
            "inside <answer>. Follow the query's required answer type/format exactly (e.g., option letter, yes/no, number, or word). "
            "Nothing else."
        )

    def _extract_response_text(self, response) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts: List[str] = []
        for output_item in getattr(response, "output", []) or []:
            for content_item in getattr(output_item, "content", []) or []:
                piece = getattr(content_item, "text", None)
                if isinstance(piece, str) and piece:
                    parts.append(piece)

        if parts:
            return "".join(parts)
        return ""

    def generate_until(self, requests) -> List[str]:
        res: List[str] = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        from lmms_eval import utils

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            task_name = task[0]
            split_name = split[0]

            batch_payloads: List[Dict[str, object]] = []
            batch_doc_uuids: List[str] = []
            batch_responses: List[Optional[str]] = []

            for i, (context, doc_id_single) in enumerate(zip(contexts, doc_id)):
                doc_uuid = f"{task_name}___{split_name}___{doc_id_single}"
                batch_doc_uuids.append(doc_uuid)

                if self.continual_mode and self.cache_mode == "resume":
                    cached = self.response_cache.get(doc_uuid)
                    if cached:
                        batch_responses.append(cached)
                        continue

                visuals = [doc_to_visual[i](self.task_dict[task_name][split_name][doc_id_single])]
                if None in visuals:
                    visuals = []
                    encoded_images: List[str] = []
                else:
                    visuals = self.flatten(visuals)
                    encoded_images = []
                    for visual in visuals:
                        if isinstance(visual, str) and (".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                            encoded_images.extend(self.encode_video(visual, self.max_frames_num))
                        else:
                            encoded_images.append(self.encode_image(visual))

                prompt_text = self._build_prompted_context(context)
                input_content: List[Dict[str, str]] = [{"type": "input_text", "text": prompt_text}]
                for image_b64 in encoded_images:
                    input_content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        }
                    )

                payload: Dict[str, object] = {
                    "model": self.model_version,
                    "input": [{"role": "user", "content": input_content}],
                }

                batch_payloads.append(payload)
                batch_responses.append(None)

            def process_single_request(payload: Dict[str, object], i: int) -> Tuple[str, int]:
                if batch_responses[i] is not None:
                    return str(batch_responses[i]), i

                for attempt in range(self.max_retries):
                    try:
                        response = self.client.responses.create(**payload)
                        response_text = self._extract_response_text(response)
                        return response_text, i
                    except Exception as exc:
                        error_msg = str(exc)
                        eval_logger.info(
                            f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}"
                        )
                        if attempt == self.max_retries - 1:
                            eval_logger.error(
                                f"All {self.max_retries} attempts failed. Last error: {error_msg}"
                            )
                            return "", i
                        time.sleep(self.timeout)
                return "", i

            tasks_to_run = [(payload, i) for i, payload in enumerate(batch_payloads) if batch_responses[i] is None]
            if tasks_to_run:
                max_workers = min(len(tasks_to_run), 32)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(process_single_request, payload, i): i
                        for payload, i in tasks_to_run
                    }
                    for future in as_completed(futures):
                        response_text, i = future.result()
                        batch_responses[i] = response_text

            if self.continual_mode:
                for doc_uuid, response_text in zip(batch_doc_uuids, batch_responses):
                    if response_text is not None:
                        self.response_cache[doc_uuid] = response_text
                with open(self.response_persistent_file, "w", encoding="utf-8") as f:
                    json.dump(self.response_cache, f)

            res.extend([r for r in batch_responses if r is not None])
            pbar.update(1)

        pbar.close()
        res = re_ords.get_original(res)
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for STTVAnswerOnlyOpenAI")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for STTVAnswerOnlyOpenAI")
