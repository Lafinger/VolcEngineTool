from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import uuid
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence
from urllib import error, request


API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"
DEFAULT_CLUSTER = "volcano_tts"  # Deprecated: V3 SSE interface does not use cluster.
DEFAULT_VOICE_TYPE = "zh_female_shuangkuaisisi_moon_bigtts"
DEFAULT_RESOURCE_ID = "volc.service_type.10029"
DEFAULT_OUTPUT_DIR_NAME = "tts_https_wavs"
DEFAULT_UID = "volcengine-tool"
DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_SAMPLE_RATE = 24000
SUCCESS_CODE = 0
SESSION_FINISH_CODE = 20000000
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


class TTSError(RuntimeError):
    """Base error for TTS operations."""


class TTSConfigError(TTSError):
    """Raised when required configuration is missing or inconsistent."""


class TTSRequestError(TTSError):
    """Raised when the remote API request fails."""


class WAVMergeError(TTSError):
    """Raised when WAV segments cannot be merged."""


@dataclass(frozen=True)
class Credentials:
    app_id: str
    access_token: str
    resource_id: str


@dataclass(frozen=True)
class MarkdownSection:
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class MarkdownFailure:
    title: str
    reason: str


@dataclass(frozen=True)
class MarkdownSynthesisResult:
    outputs: list[Path]
    failures: list[MarkdownFailure]


@lru_cache(maxsize=1)
def _load_credentials() -> Credentials:
    app_id = os.environ.get("VOLC_APP_ID", "").strip() or os.environ.get("VOLC_APP_KEY", "").strip()
    access_token = os.environ.get("VOLC_ACCESS_TOKEN", "").strip()
    resource_id = os.environ.get("VOLC_RESOURCE_ID", "").strip() or DEFAULT_RESOURCE_ID
    if not app_id or not access_token:
        raise TTSConfigError(
            "Missing credentials. Set VOLC_APP_ID (or VOLC_APP_KEY) and VOLC_ACCESS_TOKEN first."
        )
    return Credentials(
        app_id=app_id,
        access_token=access_token,
        resource_id=resource_id,
    )


def _resolve_credentials(resource_id: str | None = None) -> Credentials:
    credentials = _load_credentials()
    if resource_id and resource_id.strip():
        return Credentials(
            app_id=credentials.app_id,
            access_token=credentials.access_token,
            resource_id=resource_id.strip(),
        )
    return credentials


def _resource_not_granted_message(resource_id: str) -> str:
    return (
        "TTS resource is not granted for the current app/token. "
        f"Current resource id: {resource_id}. "
        "Use the exact resource id shown in the Volcengine console and confirm this app "
        "has been granted that TTS resource."
    )


def _is_resource_not_granted(code: object, message: str) -> bool:
    return "requested resource not granted" in message


def _validate_voice_and_resource(voice_type: str, resource_id: str) -> None:
    if voice_type.startswith("saturn_") and resource_id.startswith("seed-tts-"):
        raise TTSConfigError(
            "Current voice_type starts with 'saturn_', which usually indicates a voice-cloning "
            "or ICL 2.0 speaker. It does not match seed-tts-* public TTS resources. "
            "Use a public speaker such as zh_female_shuangkuaisisi_moon_bigtts, "
            "or switch resource_id to the matching seed-icl-2.0 resource."
        )


def _build_payload(
    text: str,
    *,
    voice_type: str,
) -> dict[str, object]:
    if not text.strip():
        raise TTSRequestError("Input text is empty after trimming.")

    return {
        "user": {
            "uid": DEFAULT_UID,
        },
        "req_params": {
            "text": text,
            "speaker": voice_type,
            "audio_params": {
                "format": "wav",
                "sample_rate": DEFAULT_SAMPLE_RATE,
            },
        },
    }


def _build_headers(credentials: Credentials, request_id: str) -> dict[str, str]:
    return {
        "X-Api-App-Id": credentials.app_id,
        "X-Api-Access-Key": credentials.access_token,
        "X-Api-Resource-Id": credentials.resource_id,
        "X-Api-Request-Id": request_id,
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
    }


def _decode_base64_audio(encoded_audio: str) -> bytes:
    try:
        return base64.b64decode(encoded_audio)
    except (TypeError, ValueError) as exc:
        raise TTSRequestError("Failed to decode base64 audio data.") from exc


def _iter_sse_payloads(response) -> Iterable[str]:
    data_lines: list[str] = []

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


def _parse_error_body(body: str, resource_id: str) -> str:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body

    if (
        isinstance(payload, dict)
        and _is_resource_not_granted(payload.get("code"), str(payload.get("message", "")))
    ):
        return _resource_not_granted_message(resource_id)

    return body


def _post_tts_request(
    payload: dict[str, object],
    credentials: Credentials,
) -> list[bytes]:
    request_id = str(uuid.uuid4())
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        API_URL,
        data=request_body,
        headers=_build_headers(credentials, request_id),
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            audio_segments: list[bytes] = []
            log_id = response.headers.get("X-Tt-Logid", "")

            for raw_payload in _iter_sse_payloads(response):
                try:
                    message = json.loads(raw_payload)
                except json.JSONDecodeError as exc:
                    raise TTSRequestError(f"Invalid SSE payload: {raw_payload}") from exc

                if not isinstance(message, dict):
                    raise TTSRequestError("Unexpected SSE payload shape from TTS API.")

                code = message.get("code")
                server_message = str(message.get("message", ""))
                encoded_audio = message.get("data")

                if code == SUCCESS_CODE:
                    if isinstance(encoded_audio, str) and encoded_audio:
                        audio_segments.append(_decode_base64_audio(encoded_audio))
                    continue

                if code == SESSION_FINISH_CODE:
                    continue

                if _is_resource_not_granted(code, server_message):
                    raise TTSRequestError(_resource_not_granted_message(credentials.resource_id))

                suffix = f" (logid: {log_id})" if log_id else ""
                raise TTSRequestError(f"TTS failed with code {code}: {server_message}{suffix}")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        parsed_body = _parse_error_body(body, credentials.resource_id)
        raise TTSRequestError(f"HTTP {exc.code}: {parsed_body}") from exc
    except error.URLError as exc:
        raise TTSRequestError(f"Request failed: {exc.reason}") from exc

    if not audio_segments:
        raise TTSRequestError("TTS stream finished without audio chunks.")

    return audio_segments


def _sanitize_filename(raw_name: str) -> str:
    sanitized = INVALID_FILENAME_CHARS.sub("_", raw_name).strip().rstrip(". ")
    sanitized = re.sub(r"\s+", " ", sanitized)
    return sanitized or "untitled"


def _parse_markdown_sections(markdown_text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            if current_title is not None and current_lines:
                sections.append(MarkdownSection(current_title, tuple(current_lines)))
            current_title = line[2:].strip()
            current_lines = []
            continue

        if current_title is None:
            continue

        stripped = line.strip()
        if stripped:
            current_lines.append(stripped)

    if current_title is not None and current_lines:
        sections.append(MarkdownSection(current_title, tuple(current_lines)))

    return sections


def _build_output_path(output_dir: Path, title: str, used_names: set[str]) -> Path:
    base_name = _sanitize_filename(title)
    candidate_name = f"{base_name}.wav"
    suffix = 2

    while candidate_name.lower() in used_names:
        candidate_name = f"{base_name}-{suffix}.wav"
        suffix += 1

    used_names.add(candidate_name.lower())
    return output_dir / candidate_name


def _read_wav_segment(segment: bytes) -> tuple[tuple[int, int, int, str, str], bytes]:
    try:
        with wave.open(io.BytesIO(segment), "rb") as reader:
            params = (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
                reader.getcomptype(),
                reader.getcompname(),
            )
            frames = reader.readframes(reader.getnframes())
    except wave.Error as exc:
        raise WAVMergeError("Received invalid WAV data from TTS API.") from exc

    return params, frames


def _merge_wav_segments(segments: Iterable[bytes]) -> bytes:
    segment_list = list(segments)
    if not segment_list:
        raise WAVMergeError("Cannot merge an empty WAV segment list.")
    if len(segment_list) == 1:
        return segment_list[0]

    expected_params: tuple[int, int, int, str, str] | None = None
    pcm_frames: list[bytes] = []

    for segment in segment_list:
        params, frames = _read_wav_segment(segment)
        if expected_params is None:
            expected_params = params
        elif params != expected_params:
            raise WAVMergeError("WAV segments have inconsistent audio parameters.")
        pcm_frames.append(frames)

    assert expected_params is not None

    merged_buffer = io.BytesIO()
    with wave.open(merged_buffer, "wb") as writer:
        writer.setnchannels(expected_params[0])
        writer.setsampwidth(expected_params[1])
        writer.setframerate(expected_params[2])
        writer.setcomptype(expected_params[3], expected_params[4])
        for frames in pcm_frames:
            writer.writeframes(frames)

    return merged_buffer.getvalue()


def _request_tts_audio_bytes(
    text: str,
    *,
    voice_type: str = DEFAULT_VOICE_TYPE,
    cluster: str = DEFAULT_CLUSTER,
    resource_id: str | None = None,
) -> bytes:
    del cluster  # Kept only for backward-compatible function signatures.
    credentials = _resolve_credentials(resource_id)
    _validate_voice_and_resource(voice_type, credentials.resource_id)
    payload = _build_payload(text, voice_type=voice_type)
    audio_segments = _post_tts_request(payload, credentials)
    return _merge_wav_segments(audio_segments)


def synthesize_text(
    text: str,
    output_path: str | Path,
    *,
    voice_type: str = DEFAULT_VOICE_TYPE,
    cluster: str = DEFAULT_CLUSTER,
    resource_id: str | None = None,
) -> Path:
    """Synthesize one text block into a WAV file."""
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    audio_bytes = _request_tts_audio_bytes(
        text,
        voice_type=voice_type,
        cluster=cluster,
        resource_id=resource_id,
    )
    target_path.write_bytes(audio_bytes)
    return target_path


def _synthesize_markdown_internal(
    markdown_path: Path,
    output_dir: Path,
    *,
    voice_type: str = DEFAULT_VOICE_TYPE,
    cluster: str = DEFAULT_CLUSTER,
    resource_id: str | None = None,
) -> MarkdownSynthesisResult:
    credentials = _resolve_credentials(resource_id)
    _validate_voice_and_resource(voice_type, credentials.resource_id)
    sections = _parse_markdown_sections(markdown_path.read_text(encoding="utf-8-sig"))
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    failures: list[MarkdownFailure] = []
    used_names: set[str] = set()

    for section in sections:
        output_path = _build_output_path(output_dir, section.title, used_names)
        try:
            segment_bytes = [
                _request_tts_audio_bytes(
                    line,
                    voice_type=voice_type,
                    cluster=cluster,
                    resource_id=credentials.resource_id,
                )
                for line in section.lines
            ]
            merged_audio = _merge_wav_segments(segment_bytes)
            output_path.write_bytes(merged_audio)
            outputs.append(output_path)
        except TTSError as exc:
            if output_path.exists():
                output_path.unlink()
            failures.append(MarkdownFailure(title=section.title, reason=str(exc)))

    return MarkdownSynthesisResult(outputs=outputs, failures=failures)


def synthesize_markdown(
    markdown_path: str | Path,
    output_dir: str | Path,
    *,
    voice_type: str = DEFAULT_VOICE_TYPE,
    cluster: str = DEFAULT_CLUSTER,
    resource_id: str | None = None,
) -> list[Path]:
    """Synthesize all level-1 markdown sections into chapter WAV files."""
    result = _synthesize_markdown_internal(
        Path(markdown_path),
        Path(output_dir),
        voice_type=voice_type,
        cluster=cluster,
        resource_id=resource_id,
    )
    return result.outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Volcengine HTTPS TTS helper.")
    parser.add_argument("--text", help="Text to synthesize into one WAV file.")
    parser.add_argument("--output", help="Output WAV path for --text mode.")
    parser.add_argument("--markdown", help="Markdown file to batch synthesize by # title.")
    parser.add_argument(
        "--voice-type",
        "--speaker",
        dest="voice_type",
        default=DEFAULT_VOICE_TYPE,
        help="Volcengine speaker id.",
    )
    parser.add_argument(
        "--cluster",
        default=DEFAULT_CLUSTER,
        help="Deprecated. V3 SSE interface ignores this argument.",
    )
    parser.add_argument(
        "--resource-id",
        default=None,
        help="Volcengine resource id. Defaults to VOLC_RESOURCE_ID or volc.service_type.10029.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    text_mode = bool(args.text)
    markdown_mode = bool(args.markdown)

    if text_mode == markdown_mode:
        parser.error("Use exactly one of --text or --markdown.")
    if text_mode and not args.output:
        parser.error("--output is required when using --text.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    try:
        if args.text:
            output_path = synthesize_text(
                args.text,
                args.output,
                voice_type=args.voice_type,
                cluster=args.cluster,
                resource_id=args.resource_id,
            )
            print(f"Generated: {output_path}")
            return 0

        markdown_path = Path(args.markdown)
        output_dir = Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIR_NAME
        result = _synthesize_markdown_internal(
            markdown_path,
            output_dir,
            voice_type=args.voice_type,
            cluster=args.cluster,
            resource_id=args.resource_id,
        )

        print(f"Success: {len(result.outputs)}")
        for output_path in result.outputs:
            print(f"  [OK] {output_path.name}")

        if result.failures:
            print(f"Failed: {len(result.failures)}", file=sys.stderr)
            for failure in result.failures:
                print(f"  [FAIL] {failure.title}: {failure.reason}", file=sys.stderr)
            return 1

        print(f"Output dir: {output_dir}")
        return 0
    except (OSError, TTSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
