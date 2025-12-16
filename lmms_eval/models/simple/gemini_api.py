import io
import json
import os
import pathlib
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - optional dependency
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    GENAI_IMPORT_ERROR = exc
else:
    GENAI_IMPORT_ERROR = None

try:
    from google.oauth2 import service_account
except ImportError:  # pragma: no cover - optional dependency
    service_account = None  # type: ignore[assignment]

try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    sf = None  # type: ignore[assignment]
    eval_logger.warning(f"Error importing soundfile, audio generation will not work: {str(e)}")

NUM_SECONDS_TO_SLEEP = 30
SCOPES = (
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language",
)
SAFETY_CATEGORY_NAMES: Sequence[str] = (
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
)
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".flv", ".wmv", ".mkv")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")


def _ensure_genai_available() -> None:
    if genai is None or genai_types is None:
        base_message = "google-genai is required for Gemini models. Install it with `pip install google-genai`."
        if GENAI_IMPORT_ERROR is not None:
            base_message = f"{base_message} (import error: {GENAI_IMPORT_ERROR})"
        raise ImportError(base_message)


def _find_service_account_file(provided: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
    if provided:
        candidates.append(Path(provided).expanduser())

    env_candidates = [
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        os.getenv("GEMINI_SERVICE_ACCOUNT_FILE"),
    ]
    for env_value in env_candidates:
        if env_value:
            candidates.append(Path(env_value).expanduser())

    candidates.append(Path.cwd() / "gemini.json")

    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append(parent / "gemini.json")

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate if candidate.is_absolute() else candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _load_service_account_credentials(
    provided: Optional[str],
    *,
    required: bool,
) -> Optional["service_account.Credentials"]:
    if service_account is None:
        raise ImportError("google-auth is required for service account authentication. Install it with `pip install google-auth`.")

    path = _find_service_account_file(provided)
    if not path:
        if required:
            raise FileNotFoundError(
                "Gemini service account credentials not found. "
                "Provide GOOGLE_APPLICATION_CREDENTIALS, GEMINI_SERVICE_ACCOUNT_FILE, "
                "or place gemini.json in the workspace.",
            )
        return None

    credentials = service_account.Credentials.from_service_account_file(str(path), scopes=SCOPES)
    eval_logger.debug(f"Loaded Gemini service account credentials from: {path}")
    return credentials


def _build_http_options(api_version: Optional[str]):
    if not api_version:
        return None
    _ensure_genai_available()
    if not hasattr(genai_types, "HttpOptions"):
        raise AttributeError("google-genai.types.HttpOptions is not available in the installed version.")
    return genai_types.HttpOptions(api_version=api_version)


def _create_genai_client(
    *,
    api_key: Optional[str],
    vertexai: bool,
    project: Optional[str],
    location: Optional[str],
    service_account_file: Optional[str],
    http_api_version: Optional[str],
) -> "genai.Client":
    _ensure_genai_available()

    client_kwargs: dict[str, Any] = {}

    http_options = _build_http_options(http_api_version)
    if http_options is not None:
        client_kwargs["http_options"] = http_options

    credentials_required = vertexai or bool(service_account_file) or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GEMINI_SERVICE_ACCOUNT_FILE")
    credentials = _load_service_account_credentials(service_account_file, required=bool(credentials_required))
    if credentials is not None:
        client_kwargs["credentials"] = credentials

    if vertexai:
        client_kwargs["vertexai"] = True
        resolved_project = project or os.getenv("GOOGLE_VERTEX_PROJECT")
        if not resolved_project:
            raise ValueError("Vertex AI project is required. Provide `vertex_project` or set GOOGLE_VERTEX_PROJECT.")
        client_kwargs["project"] = resolved_project
        client_kwargs["location"] = location or os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
    else:
        if credentials is None:
            resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError("Provide GOOGLE_API_KEY or service account credentials to authenticate with Gemini.")
            client_kwargs["api_key"] = resolved_key

    return genai.Client(**client_kwargs)


def _default_safety_settings():
    _ensure_genai_available()
    if not hasattr(genai_types, "SafetySetting"):
        return None
    settings = []
    for category in SAFETY_CATEGORY_NAMES:
        try:
            settings.append(genai_types.SafetySetting(category=category, threshold="BLOCK_NONE"))
        except Exception as exc:  # pragma: no cover - best effort
            eval_logger.debug(f"Failed to configure safety setting for {category}: {exc}")
    return settings or None


def _ensure_part_from_image(image: Image.Image):
    _ensure_genai_available()
    if hasattr(genai_types.Part, "from_image"):
        return genai_types.Part.from_image(image=image)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    data = buffer.getvalue()
    if hasattr(genai_types.Part, "from_bytes"):
        return genai_types.Part.from_bytes(data=data, mime_type="image/png")  # type: ignore[attr-defined]
    if hasattr(genai_types.Part, "inline_data"):
        return genai_types.Part(inline_data=genai_types.Part.InlineData(data=data, mime_type="image/png"))  # type: ignore[attr-defined]
    raise AttributeError("Unable to construct an image part with the installed google-genai library.")


def _ensure_part_from_file(uploaded_file):
    _ensure_genai_available()
    if hasattr(genai_types.Part, "from_file"):
        return genai_types.Part.from_file(file=uploaded_file)
    if hasattr(uploaded_file, "uri"):
        return genai_types.Part(file_data=genai_types.Part.FileData(file_uri=uploaded_file.uri))  # type: ignore[attr-defined]
    raise ValueError("Uploaded file type is not supported by the installed google-genai library.")


def _ensure_part_from_text(text: str):
    _ensure_genai_available()
    return genai_types.Part(text=text)


def _coerce_to_part(item):
    if item is None:
        return None
    if isinstance(item, str):
        stripped = item.strip()
        if not stripped:
            return None
        return _ensure_part_from_text(stripped)
    if isinstance(item, Image.Image):
        return _ensure_part_from_image(item)
    if genai_types is not None and hasattr(genai_types, "File") and isinstance(item, genai_types.File):  # type: ignore[attr-defined]
        return _ensure_part_from_file(item)
    if hasattr(item, "name") and hasattr(item, "uri"):  # likely a File instance
        return _ensure_part_from_file(item)
    eval_logger.warning(f"Unsupported modality item type {type(item)}; skipping.")
    return None


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        timeout: int = 120,
        continual_mode: bool = True,
        response_persistent_folder: str = "./logs/gemini_persistent_folder",
        interleave: bool = False,
        api_key: Optional[str] = None,
        vertexai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        service_account_file: Optional[str] = None,
        http_api_version: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        _ensure_genai_available()

        self.model_version = model_version
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.response_persistent_file = ""
        self.interleave = interleave

        self.client = _create_genai_client(
            api_key=api_key,
            vertexai=vertexai,
            project=vertex_project,
            location=vertex_location,
            service_account_file=service_account_file,
            http_api_version=http_api_version,
        )
        self.safety_settings = _default_safety_settings()
        self._uploaded_files: List[Any] = []

        self.response_cache: dict[str, Any] = {}
        self.cache_mode = "start"

        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            os.makedirs(self.response_persistent_folder, exist_ok=True)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r", encoding="utf-8") as f:
                    try:
                        self.response_cache = json.load(f)
                        self.cache_mode = "resume"
                    except json.JSONDecodeError:
                        eval_logger.warning(f"Persistent cache file {self.response_persistent_file} is malformed; starting fresh.")
                        self.response_cache = {}
                        self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def free_video(self):
        if not self._uploaded_files:
            return
        for uploaded in self._uploaded_files:
            try:
                file_name = getattr(uploaded, "name", None)
                if file_name:
                    self.client.files.delete(name=file_name)
            except Exception as exc:  # pragma: no cover - best effort cleanup
                eval_logger.debug(f"Failed to delete uploaded file {getattr(uploaded, 'name', 'unknown')}: {exc}")
        self._uploaded_files = []

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def encode_video(self, video_path):
        uploaded_obj = self.client.files.upload(file=video_path)
        # Allow some time for the backend to process large uploads when needed.
        time.sleep(2)
        self._uploaded_files.append(uploaded_obj)
        return uploaded_obj

    def encode_audio(self, audio):
        if sf is None:
            raise ImportError("soundfile is required for audio inputs. Install it with `pip install soundfile`.")
        audio_io = io.BytesIO()
        sf.write(audio_io, audio["array"], audio["sampling_rate"], format="WAV")
        audio_io.seek(0)
        uploaded_audio = self.client.files.upload(file=audio_io, mime_type="audio/wav", display_name="audio.wav")
        self._uploaded_files.append(uploaded_audio)
        return uploaded_audio

    def convert_modality(self, images):
        converted = list(images)
        for idx, img in enumerate(converted):
            if isinstance(img, dict) and "sampling_rate" in img:  # audio
                try:
                    converted[idx] = self.encode_audio(img)
                except Exception as exc:
                    eval_logger.error(f"Error converting audio: {str(exc)}")
                    converted[idx] = None
            elif isinstance(img, str):
                lower = img.lower()
                if lower.endswith(VIDEO_EXTENSIONS):
                    try:
                        converted[idx] = self.encode_video(img)
                    except Exception as exc:
                        eval_logger.error(f"Error converting video {img}: {str(exc)}")
                        converted[idx] = None
                elif lower.endswith(IMAGE_EXTENSIONS):
                    try:
                        converted[idx] = Image.open(img)
                    except Exception as exc:
                        eval_logger.error(f"Error opening image {img}: {str(exc)}")
                        converted[idx] = None
        return converted

    def construct_interleaved_input(self, content, media):
        pattern = r"<media_(\d+)>"
        parts = re.split(pattern, content)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part == "":
                    continue
                result.append(part)
            else:
                result.append(media[int(part)])

        return result

    def _build_user_parts(self, context: str, visuals: List[Any]):
        if self.interleave:
            sequence = self.construct_interleaved_input(context, visuals)
        else:
            sequence = [context] + visuals

        parts = []
        for item in sequence:
            part = _coerce_to_part(item)
            if part is not None:
                parts.append(part)

        if not parts:
            parts.append(_ensure_part_from_text(context))
        return parts

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0

            config_kwargs = {
                "max_output_tokens": gen_kwargs.get("max_new_tokens"),
                "temperature": gen_kwargs.get("temperature"),
            }
            if gen_kwargs.get("top_p") is not None:
                config_kwargs["top_p"] = gen_kwargs["top_p"]

            config_obj: Optional[Any] = None
            cleaned_config = {k: v for k, v in config_kwargs.items() if v is not None}
            if cleaned_config:
                if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
                    config_obj = genai_types.GenerateContentConfig(**cleaned_config)
                elif genai_types is not None and hasattr(genai_types, "GenerationConfig"):
                    config_obj = genai_types.GenerationConfig(**cleaned_config)
                else:
                    config_obj = cleaned_config

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            visuals = self.convert_modality(visuals)

            user_parts = self._build_user_parts(contexts, visuals)
            user_content = None
            if genai_types is not None and hasattr(genai_types, "Content"):
                user_content = genai_types.Content(role="user", parts=user_parts)

            content = ""
            for attempt in range(5):
                try:
                    request_kwargs = {
                        "model": self.model_version,
                        "contents": [user_content] if user_content is not None else user_parts,
                    }
                    if config_obj is not None:
                        request_kwargs["config"] = config_obj
                    if self.safety_settings is not None:
                        request_kwargs["safety_settings"] = self.safety_settings

                    response = self.client.models.generate_content(**request_kwargs)
                    content_text = getattr(response, "text", None)
                    if content_text is None and hasattr(response, "candidates"):
                        text_chunks = []
                        for candidate in getattr(response, "candidates", []):
                            candidate_content = getattr(candidate, "content", None)
                            candidate_parts = getattr(candidate_content, "parts", []) if candidate_content else []
                            for part in candidate_parts:
                                text_value = getattr(part, "text", None)
                                if text_value:
                                    text_chunks.append(text_value)
                        content_text = "".join(text_chunks)
                    content = content_text or ""
                    break
                except TypeError as e:
                    message = str(e)
                    if "config" in message and config_obj is not None:
                        eval_logger.warning("`config` parameter not supported by current google-genai client; retrying without config.")
                        config_obj = None
                        continue
                    if "safety_settings" in message and self.safety_settings is not None:
                        eval_logger.warning("`safety_settings` parameter not supported by current google-genai client; retrying without safety settings.")
                        self.safety_settings = None
                        continue
                    eval_logger.info(f"Attempt {attempt + 1} failed with TypeError: {message}")
                    if attempt < 5 - 1:
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                        continue
                    eval_logger.error(f"All 5 attempts failed. Last error message: {message}")
                    content = ""
                except Exception as e:
                    eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                            content = ""
                            break
                        except Exception:
                            pass
                    if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                        content = ""
            res.append(content)
            pbar.update(1)

            self.free_video()

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w", encoding="utf-8") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for Gemini API")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"

    def get_image_audio_text_interleaved_messsage(self, image_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed image token and no audio in text
        for index in range(1, 1 + len(image_path)):
            question = question.replace(f"[img{index}]", "<image>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        image_counter = 0
        audio_counter = 0
        for part in re.split(r"(<image>|<audio>)", text):
            if part == "<image>":
                info_list.append(Image.open(image_path[image_counter]))
                image_counter += 1
            elif part == "<audio>":
                audio_file_path = pathlib.Path(audio_path[audio_counter])
                try:
                    uploaded_audio = self.client.files.upload(file=str(audio_file_path), mime_type="audio/wav", display_name=audio_file_path.name)
                    self._uploaded_files.append(uploaded_audio)
                    info_list.append(uploaded_audio)
                except Exception as exc:
                    eval_logger.error(f"Failed to upload audio {audio_file_path}: {exc}")
                    info_list.append(None)
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list

    def get_video_audio_text_interleaved_message(self, video_path, audio_path, question):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed video token and no audio in text
        for index in range(1, 1 + len(video_path)):
            question = question.replace(f"[video{index}]", "<video>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        video_counter = 0
        audio_counter = 0
        for part in re.split(r"(<video>|<audio>)", text):
            if part == "<video>":
                current_video_file_name = video_path[video_counter]
                try:
                    current_video_file = self.encode_video(current_video_file_name)
                    info_list.append(current_video_file)
                except Exception as exc:
                    eval_logger.error(f"Failed to upload video {current_video_file_name}: {exc}")
                    info_list.append(None)
                video_counter += 1
            elif part == "<audio>":
                audio_file_path = pathlib.Path(audio_path[audio_counter])
                try:
                    uploaded_audio = self.client.files.upload(file=str(audio_file_path), mime_type="audio/wav", display_name=audio_file_path.name)
                    self._uploaded_files.append(uploaded_audio)
                    info_list.append(uploaded_audio)
                except Exception as exc:
                    eval_logger.error(f"Failed to upload audio {audio_file_path}: {exc}")
                    info_list.append(None)
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list
