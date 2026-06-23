"""
Media Understanding Module
==========================

Converts images, audio, and video to text descriptions using Together AI APIs.

Supported:
- Images: Llama 4 Scout vision model
- Audio: Whisper Large v3
- Video: Frame extraction + vision model (basic support)
"""

import base64
import httpx
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

# Try to import optional video processing library
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class MediaConfig:
    """Media understanding configuration"""
    enabled: bool = False

    # Together AI settings
    api_key: Optional[str] = None
    api_key_env: str = "TOGETHER_API_KEY"
    base_url: str = "https://api.together.xyz/v1"

    # Vision model for images/video frames
    vision_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

    # Audio transcription model
    audio_model: str = "openai/whisper-large-v3"

    # Prompts
    image_prompt: str = "Describe this image concisely in 2-3 sentences."
    video_prompt: str = "Describe what you see in these video frames."

    # Video settings
    video_max_frames: int = 4  # Max frames to extract from video

    # Max content length in memory
    max_description_chars: int = 500


# Supported MIME types
IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/heic", "image/heif"}
AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/m4a", "audio/mp4", "audio/webm"}
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/mpeg"}


def _get_api_key(cfg: MediaConfig, fallback_key: Optional[str] = None) -> Optional[str]:
    """Get API key from config, environment, or fallback"""
    if cfg.api_key:
        return cfg.api_key
    env_key = os.getenv(cfg.api_key_env)
    if env_key:
        return env_key
    return fallback_key


def _detect_media_from_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Detect media from plain text with [media attached: path (mime) | url] format.

    This handles the OpenClaw format where media is embedded as text placeholders.

    Returns: (media_type, file_path, mime_type, url)
    """
    import re

    # Pattern: [media attached: /path/to/file (mime/type) | optional_url]
    # Capture both path and optional URL
    pattern = r'\[media attached:\s*([^\s\(\)]+)\s*\(([^)]+)\)(?:\s*\|\s*(https?://[^\]\s]+))?'
    match = re.search(pattern, text)

    if not match:
        return None, None, None, None

    file_path = match.group(1).strip()
    mime_type = match.group(2).strip()
    url = match.group(3).strip() if match.group(3) else None

    # Determine media type from MIME
    if mime_type in IMAGE_TYPES or mime_type.startswith("image/"):
        return "image", file_path, mime_type, url
    elif mime_type in AUDIO_TYPES or mime_type.startswith("audio/"):
        return "audio", file_path, mime_type, url
    elif mime_type in VIDEO_TYPES or mime_type.startswith("video/"):
        return "video", file_path, mime_type, url

    return None, None, None, None


def _load_file_as_base64(file_path: str) -> Optional[str]:
    """Load a local file and return as base64 string."""
    try:
        # Expand ~ and resolve path
        path = Path(os.path.expanduser(file_path))
        if not path.exists():
            return None
        with open(path, "rb") as f:
            data = f.read()
            # Validate it's actually binary media, not HTML/text
            if data[:15].startswith(b'<!DOCTYPE') or data[:6].startswith(b'<html'):
                return None
            return base64.b64encode(data).decode("ascii")
    except Exception:
        return None


def _detect_media_type(content: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Detect media type from multimodal content.

    Returns: (media_type, data_or_url, mime_type)
    - media_type: "image", "audio", "video", or None
    - data_or_url: base64 data or URL string
    - mime_type: MIME type if detected
    """
    if not isinstance(content, list):
        return None, None, None

    for part in content:
        if not isinstance(part, dict):
            continue

        part_type = part.get("type", "")

        # Image URL format (OpenAI style)
        if part_type == "image_url":
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            if url.startswith("data:"):
                # data:image/jpeg;base64,/9j/4AAQ...
                try:
                    header, b64_data = url.split(",", 1)
                    mime = header.split(";")[0].replace("data:", "")
                    if mime in IMAGE_TYPES:
                        return "image", b64_data, mime
                except ValueError:
                    pass
            elif url.startswith("http"):
                return "image", url, None

        # Direct image format
        elif part_type == "image":
            data = part.get("data", "")
            mime = part.get("mimeType", part.get("mime_type", "image/jpeg"))
            if data:
                return "image", data, mime

        # Audio format
        elif part_type == "audio" or part_type == "input_audio":
            data = part.get("data", "")
            mime = part.get("mimeType", part.get("mime_type", "audio/mp3"))
            if data:
                return "audio", data, mime

        # Video format
        elif part_type == "video":
            data = part.get("data", "")
            mime = part.get("mimeType", part.get("mime_type", "video/mp4"))
            if data:
                return "video", data, mime

    return None, None, None


async def describe_image(
    data_or_url: str,
    cfg: MediaConfig,
    mime_type: Optional[str] = None,
    is_url: bool = False,
    fallback_key: Optional[str] = None,
) -> str:
    """
    Get text description of an image using Together AI vision model.

    Args:
        data_or_url: Base64 encoded image data or URL
        cfg: Media configuration
        mime_type: MIME type (default: image/jpeg)
        is_url: True if data_or_url is a URL
        fallback_key: Fallback API key if not in config/env

    Returns:
        Text description of the image
    """
    api_key = _get_api_key(cfg, fallback_key)
    if not api_key:
        return "[Image: API key not configured]"

    mime = mime_type or "image/jpeg"

    # Build image content
    if is_url:
        image_content = {"type": "image_url", "image_url": {"url": data_or_url}}
    else:
        data_url = f"data:{mime};base64,{data_or_url}"
        image_content = {"type": "image_url", "image_url": {"url": data_url}}

    body = {
        "model": cfg.vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cfg.image_prompt},
                    image_content,
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.3,
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{cfg.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=60.0,
            )

            if resp.status_code != 200:
                return f"[Image: Vision API error {resp.status_code}]"

            result = resp.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Truncate if needed
            if cfg.max_description_chars > 0 and len(content) > cfg.max_description_chars:
                content = content[:cfg.max_description_chars] + "..."

            return f"[Image: {content}]" if content else "[Image: No description available]"

    except Exception as e:
        return f"[Image: Error - {str(e)[:100]}]"


async def transcribe_audio(
    data: str,
    cfg: MediaConfig,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
    fallback_key: Optional[str] = None,
) -> str:
    """
    Transcribe audio to text using Together AI Whisper.

    Args:
        data: Base64 encoded audio data
        cfg: Media configuration
        mime_type: MIME type (default: audio/mp3)
        language: Language code (optional, auto-detected if not specified)
        fallback_key: Fallback API key if not in config/env

    Returns:
        Transcribed text
    """
    api_key = _get_api_key(cfg, fallback_key)
    if not api_key:
        return "[Audio: API key not configured]"

    mime = mime_type or "audio/mp3"
    ext_map = {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/ogg": ".ogg",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/webm": ".webm",
    }
    ext = ext_map.get(mime, ".mp3")

    try:
        # Decode base64 and save to temp file
        audio_bytes = base64.b64decode(data)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            async with httpx.AsyncClient() as client:
                # Whisper API uses multipart form data
                files = {"file": (f"audio{ext}", open(tmp_path, "rb"), mime)}
                data_fields = {"model": cfg.audio_model}
                if language:
                    data_fields["language"] = language

                resp = await client.post(
                    f"{cfg.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files=files,
                    data=data_fields,
                    timeout=120.0,
                )

                if resp.status_code != 200:
                    return f"[Audio: Transcription API error {resp.status_code}]"

                result = resp.json()
                text = result.get("text", "")

                # Truncate if needed
                if cfg.max_description_chars > 0 and len(text) > cfg.max_description_chars:
                    text = text[:cfg.max_description_chars] + "..."

                return f"[Audio transcript: {text}]" if text else "[Audio: No speech detected]"

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return f"[Audio: Error - {str(e)[:100]}]"


async def describe_video(
    data: str,
    cfg: MediaConfig,
    mime_type: Optional[str] = None,
    fallback_key: Optional[str] = None,
) -> str:
    """
    Get text description of a video by extracting key frames.

    Args:
        data: Base64 encoded video data
        cfg: Media configuration
        mime_type: MIME type (default: video/mp4)
        fallback_key: Fallback API key if not in config/env

    Returns:
        Text description of the video
    """
    if not HAS_CV2:
        return "[Video: opencv-python not installed, cannot process video]"

    api_key = _get_api_key(cfg, fallback_key)
    if not api_key:
        return "[Video: API key not configured]"

    mime = mime_type or "video/mp4"
    ext_map = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/mpeg": ".mpeg",
    }
    ext = ext_map.get(mime, ".mp4")

    try:
        # Decode base64 and save to temp file
        video_bytes = base64.b64decode(data)

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            # Extract frames
            frames = _extract_video_frames(tmp_path, cfg.video_max_frames)

            if not frames:
                return "[Video: Could not extract frames]"

            # Build multimodal content with all frames
            content_parts = [{"type": "text", "text": cfg.video_prompt}]
            for frame_b64 in frames:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                })

            body = {
                "model": cfg.vision_model,
                "messages": [{"role": "user", "content": content_parts}],
                "max_tokens": 256,
                "temperature": 0.3,
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{cfg.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=90.0,
                )

                if resp.status_code != 200:
                    return f"[Video: Vision API error {resp.status_code}]"

                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Truncate if needed
                if cfg.max_description_chars > 0 and len(content) > cfg.max_description_chars:
                    content = content[:cfg.max_description_chars] + "..."

                return f"[Video: {content}]" if content else "[Video: No description available]"

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        return f"[Video: Error - {str(e)[:100]}]"


def _extract_video_frames(video_path: str, max_frames: int = 4) -> List[str]:
    """
    Extract key frames from video and return as base64 JPEG strings.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract

    Returns:
        List of base64 encoded JPEG frames
    """
    if not HAS_CV2:
        return []

    frames = []
    cap = cv2.VideoCapture(video_path)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return []

        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= max_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            indices = [i * step for i in range(max_frames)]

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize if too large (max 1024px on longest side)
            h, w = frame.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h))

            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            b64 = base64.b64encode(buffer).decode("ascii")
            frames.append(b64)

    finally:
        cap.release()

    return frames


async def process_multimodal_content(
    content: Any,
    cfg: MediaConfig,
    fallback_key: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Process multimodal content and return text representation.

    Supports two formats:
    1. OpenAI multimodal format: content as list with {"type": "image_url", ...}
    2. OpenClaw text format: "[media attached: /path/to/file (mime/type) | url]"

    Args:
        content: Message content (string or list of parts)
        cfg: Media configuration
        fallback_key: Fallback API key (e.g., from api_keys.together)

    Returns:
        Tuple of (processed_text, media_description)
        - processed_text: Original text + media description
        - media_description: Just the media description (for memory)
    """
    if not cfg.enabled:
        if isinstance(content, str):
            return content, None
        # Extract text from list
        text_parts = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
        return "\n".join(text_parts).strip(), None

    # Handle plain text with [media attached: ...] format (OpenClaw format)
    if isinstance(content, str):
        media_type, file_path, mime_type, media_url = _detect_media_from_text(content)

        if not media_type or not file_path:
            return content, None

        # Load file as base64
        b64_data = _load_file_as_base64(file_path)

        # If local file failed and we have a URL, try URL-based processing
        if not b64_data and media_url:
            # For images, we can pass URL directly to vision API
            if media_type == "image":
                media_desc = await describe_image(media_url, cfg, mime_type, is_url=True, fallback_key=fallback_key)
                if media_desc and not media_desc.startswith("[Image: Vision API error"):
                    import re
                    clean_text = re.sub(r'\[media attached:[^\]]+\]', '', content).strip()
                    if clean_text:
                        combined = f"{clean_text}\n\n{media_desc}"
                    else:
                        combined = media_desc
                    return combined, media_desc

        if not b64_data:
            return content, f"[{media_type.title()}: Could not load file {file_path}]"

        # Process based on media type
        media_desc = None

        if media_type == "image":
            media_desc = await describe_image(b64_data, cfg, mime_type, False, fallback_key)
        elif media_type == "audio":
            media_desc = await transcribe_audio(b64_data, cfg, mime_type, None, fallback_key)
        elif media_type == "video":
            media_desc = await describe_video(b64_data, cfg, mime_type, fallback_key)

        if media_desc:
            # Remove the [media attached: ...] placeholder and add description
            import re
            clean_text = re.sub(r'\[media attached:[^\]]+\]', '', content).strip()
            if clean_text:
                combined = f"{clean_text}\n\n{media_desc}"
            else:
                combined = media_desc
            return combined, media_desc

        return content, None

    # Handle OpenAI multimodal format (list of parts)
    text_parts = []
    media_type, data_or_url, mime_type = None, None, None

    if isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))

    original_text = "\n".join(text_parts).strip()

    # First check if extracted text contains [media attached: ...] format (OpenClaw style)
    text_media_type, text_file_path, text_mime_type, text_media_url = _detect_media_from_text(original_text)

    if text_media_type and text_file_path:
        # Load file as base64
        b64_data = _load_file_as_base64(text_file_path)

        # If local file failed and we have a URL, try URL-based processing
        if not b64_data and text_media_url:
            if text_media_type == "image":
                media_desc = await describe_image(text_media_url, cfg, text_mime_type, is_url=True, fallback_key=fallback_key)
                if media_desc and not media_desc.startswith("[Image: Vision API error"):
                    import re
                    clean_text = re.sub(r'\[media attached:[^\]]+\]', '', original_text).strip()
                    if clean_text:
                        combined = f"{clean_text}\n\n{media_desc}"
                    else:
                        combined = media_desc
                    return combined, media_desc

        if not b64_data:
            return original_text, f"[{text_media_type.title()}: Could not load file {text_file_path}]"

        # Process based on media type
        media_desc = None

        if text_media_type == "image":
            media_desc = await describe_image(b64_data, cfg, text_mime_type, False, fallback_key)
        elif text_media_type == "audio":
            media_desc = await transcribe_audio(b64_data, cfg, text_mime_type, None, fallback_key)
        elif text_media_type == "video":
            media_desc = await describe_video(b64_data, cfg, text_mime_type, fallback_key)

        if media_desc:
            # Remove the [media attached: ...] placeholder and add description
            import re
            clean_text = re.sub(r'\[media attached:[^\]]+\]', '', original_text).strip()
            if clean_text:
                combined = f"{clean_text}\n\n{media_desc}"
            else:
                combined = media_desc
            return combined, media_desc

    # Then check for OpenAI multimodal format
    media_type, data_or_url, mime_type = _detect_media_type(content)

    if not media_type or not data_or_url:
        return original_text, None

    # Process based on media type
    media_desc = None

    if media_type == "image":
        is_url = data_or_url.startswith("http")
        media_desc = await describe_image(data_or_url, cfg, mime_type, is_url, fallback_key)

    elif media_type == "audio":
        media_desc = await transcribe_audio(data_or_url, cfg, mime_type, None, fallback_key)

    elif media_type == "video":
        media_desc = await describe_video(data_or_url, cfg, mime_type, fallback_key)

    if media_desc:
        # Combine text with media description
        if original_text:
            combined = f"{original_text}\n\n{media_desc}"
        else:
            combined = media_desc
        return combined, media_desc

    return original_text, None
