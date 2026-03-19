# -*- coding: utf-8 -*-
"""火山引擎豆包语音合成 HTTP Chunked 最小实现。

PowerShell 环境变量示例：
    $env:VOLCENGINE_TTS_APP_ID = "your-app-id"
    $env:VOLCENGINE_TTS_ACCESS_KEY = "your-access-key"
    $env:VOLCENGINE_TTS_UID = "demo-user"

单条文本示例：
    uv run python tts/tts_https.py v1 --text "您好，欢迎光临。" --output tts/tts_https_wavs/v1/demo.wav
    uv run python tts/tts_https.py v2 --text "您好，欢迎光临。" --output tts/tts_https_wavs/v2/demo.wav

批量导出示例：
    uv run python tts/tts_https.py batch --model v1
    uv run python tts/tts_https.py batch --model v2
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

API_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DEFAULT_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
RESOURCE_ID_V1 = "seed-tts-1.0"
RESOURCE_ID_V2 = "seed-tts-2.0"
SPEAKER_V1 = "ICL_zh_female_yry_tob"
SPEAKER_V2 = "saturn_zh_female_qingyingduoduo_cs_tob"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_INPUT = SCRIPT_DIR / "需合成的文本_中文.md"
DEFAULT_BATCH_OUTPUT = SCRIPT_DIR / "tts_https_wavs"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


class VolcengineTTSError(RuntimeError):
    """火山引擎 TTS 请求失败。"""

    def __init__(self, message: str, *, code: int | None = None, logid: str | None = None) -> None:
        self.code = code
        self.logid = logid

        parts = [message]
        if code is not None:
            parts.append(f"code={code}")
        if logid:
            parts.append(f"logid={logid}")

        super().__init__(" | ".join(parts))


@dataclass(frozen=True)
class _Credentials:
    app_id: str
    access_key: str
    uid: str


@dataclass(frozen=True)
class MarkdownSection:
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class _ModelConfig:
    model: str
    resource_id: str
    speaker: str


def _load_credentials() -> _Credentials:
    app_id = os.getenv("VOLCENGINE_TTS_APP_ID", "").strip()
    access_key = os.getenv("VOLCENGINE_TTS_ACCESS_KEY", "").strip()
    uid = os.getenv("VOLCENGINE_TTS_UID", "codex-tts").strip() or "codex-tts"

    missing = [
        name
        for name, value in (
            ("VOLCENGINE_TTS_APP_ID", app_id),
            ("VOLCENGINE_TTS_ACCESS_KEY", access_key),
        )
        if not value
    ]
    if missing:
        raise VolcengineTTSError(f"缺少环境变量：{', '.join(missing)}")

    return _Credentials(app_id=app_id, access_key=access_key, uid=uid)


def _get_model_config(model: str) -> _ModelConfig:
    if model == "v1":
        return _ModelConfig(model="v1", resource_id=RESOURCE_ID_V1, speaker=SPEAKER_V1)
    if model == "v2":
        return _ModelConfig(model="v2", resource_id=RESOURCE_ID_V2, speaker=SPEAKER_V2)
    raise ValueError(f"不支持的模型：{model}")


def _build_headers(credentials: _Credentials, resource_id: str) -> dict[str, str]:
    return {
        "X-Api-App-Id": credentials.app_id,
        "X-Api-Access-Key": credentials.access_key,
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }


def _build_payload(credentials: _Credentials, text: str, speaker: str, sample_rate: int) -> dict[str, object]:
    return {
        "user": {"uid": credentials.uid},
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {
                "format": "pcm",
                "sample_rate": sample_rate,
            },
            "additions": json.dumps({"disable_markdown_filter": True}, ensure_ascii=False),
        },
    }


def _normalize_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("text 不能为空")
    return normalized


def _synthesize_pcm(
    text: str,
    *,
    credentials: _Credentials,
    resource_id: str,
    speaker: str,
    sample_rate: int,
    session: requests.Session,
) -> bytes:
    normalized_text = _normalize_text(text)
    response: requests.Response | None = None
    logid: str | None = None

    try:
        response = session.post(
            API_URL,
            headers=_build_headers(credentials, resource_id),
            json=_build_payload(credentials, normalized_text, speaker, sample_rate),
            stream=True,
            timeout=(10, 300),
        )
        logid = response.headers.get("X-Tt-Logid")

        if response.status_code != 200:
            body = response.text.strip()
            if len(body) > 300:
                body = f"{body[:300]}..."
            raise VolcengineTTSError(
                f"HTTP 请求失败：status={response.status_code} body={body or '<empty>'}",
                logid=logid,
            )

        audio_chunks = bytearray()
        finished = False

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8").strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VolcengineTTSError("解析服务端流式响应失败", logid=logid) from exc

            code = int(event.get("code", 0) or 0)
            chunk = event.get("data")
            if code == 0 and chunk:
                try:
                    audio_chunks.extend(base64.b64decode(chunk, validate=True))
                except (binascii.Error, ValueError) as exc:
                    raise VolcengineTTSError("音频分片 Base64 解码失败", logid=logid) from exc
                continue

            if code == 0 and "sentence" in event:
                continue

            if code == 20000000:
                finished = True
                break

            raise VolcengineTTSError(
                event.get("message") or "语音合成失败",
                code=code,
                logid=logid,
            )

        if not finished:
            raise VolcengineTTSError("服务端未返回结束标记", logid=logid)

        if not audio_chunks:
            raise VolcengineTTSError("服务端未返回音频数据", logid=logid)

        if len(audio_chunks) % PCM_SAMPLE_WIDTH != 0:
            raise VolcengineTTSError("返回的 PCM 数据长度异常", logid=logid)

        return bytes(audio_chunks)
    except requests.RequestException as exc:
        raise VolcengineTTSError("请求火山引擎 TTS 接口失败", logid=logid) from exc
    finally:
        if response is not None:
            response.close()


def _write_wav(pcm_data: bytes, output_path: str | Path, sample_rate: int) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(PCM_CHANNELS)
        wav_file.setsampwidth(PCM_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return target


def _synthesize_to_wav(
    text: str,
    output_path: str | Path,
    *,
    resource_id: str,
    speaker: str,
    sample_rate: int,
    credentials: _Credentials | None = None,
    session: requests.Session | None = None,
) -> Path:
    own_session = session is None
    credentials = credentials or _load_credentials()
    active_session = session or requests.Session()

    try:
        pcm_data = _synthesize_pcm(
            text,
            credentials=credentials,
            resource_id=resource_id,
            speaker=speaker,
            sample_rate=sample_rate,
            session=active_session,
        )
    finally:
        if own_session:
            active_session.close()

    return _write_wav(pcm_data, output_path, sample_rate)


def synthesize_tts_v1(
    text: str,
    output_path: str | Path,
    *,
    speaker: str = SPEAKER_V1,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    return _synthesize_to_wav(
        text,
        output_path,
        resource_id=RESOURCE_ID_V1,
        speaker=speaker,
        sample_rate=sample_rate,
    )


def synthesize_tts_v2(
    text: str,
    output_path: str | Path,
    *,
    speaker: str = SPEAKER_V2,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    return _synthesize_to_wav(
        text,
        output_path,
        resource_id=RESOURCE_ID_V2,
        speaker=speaker,
        sample_rate=sample_rate,
    )


def parse_markdown_sections(markdown_path: str | Path) -> list[MarkdownSection]:
    text = Path(markdown_path).read_text(encoding="utf-8")
    sections: list[MarkdownSection] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("# "):
            if current_title and current_lines:
                sections.append(MarkdownSection(title=current_title, lines=tuple(current_lines)))
            current_title = stripped[2:].strip()
            current_lines = []
            continue

        if not current_title or not stripped:
            continue

        current_lines.append(stripped)

    if current_title and current_lines:
        sections.append(MarkdownSection(title=current_title, lines=tuple(current_lines)))

    return sections


def _sanitize_filename(title: str) -> str:
    sanitized = INVALID_FILENAME_CHARS.sub("_", title).strip().rstrip(". ")
    return sanitized or "untitled"


def _section_output_path(directory: Path, title: str) -> Path:
    return directory / f"{_sanitize_filename(title)}.wav"


def _combine_section_pcm(
    lines: Iterable[str],
    *,
    credentials: _Credentials,
    resource_id: str,
    speaker: str,
    sample_rate: int,
    session: requests.Session,
) -> bytes:
    audio_parts = [
        _synthesize_pcm(
            line,
            credentials=credentials,
            resource_id=resource_id,
            speaker=speaker,
            sample_rate=sample_rate,
            session=session,
        )
        for line in lines
    ]

    combined = b"".join(audio_parts)
    if not combined:
        raise VolcengineTTSError("章节未生成任何音频数据")
    return combined


def _batch_generate_model(
    sections: list[MarkdownSection],
    *,
    output_dir: Path,
    resource_id: str,
    speaker: str,
    sample_rate: int,
    credentials: _Credentials,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    total = len(sections)

    with requests.Session() as session:
        for index, section in enumerate(sections, start=1):
            output_path = _section_output_path(output_dir, section.title)
            if output_path.exists():
                print(f"[{index}/{total}] 跳过已存在：{section.title} -> {output_path}", flush=True)
                continue

            print(f"[{index}/{total}] 开始生成：{section.title} -> {output_path}", flush=True)
            try:
                pcm_data = _combine_section_pcm(
                    section.lines,
                    credentials=credentials,
                    resource_id=resource_id,
                    speaker=speaker,
                    sample_rate=sample_rate,
                    session=session,
                )
                generated_path = _write_wav(pcm_data, output_path, sample_rate)
            except Exception:
                print(f"[{index}/{total}] 生成失败：{section.title}", flush=True)
                raise

            generated_paths.append(generated_path)
            print(f"[{index}/{total}] 生成完成：{generated_path}", flush=True)

    return generated_paths


def synthesize_markdown_cases(
    markdown_path: str | Path = DEFAULT_BATCH_INPUT,
    output_root: str | Path = DEFAULT_BATCH_OUTPUT,
    *,
    model: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> list[Path]:
    sections = parse_markdown_sections(markdown_path)
    if not sections:
        raise ValueError(f"未在 {Path(markdown_path)} 中解析到可合成内容")

    credentials = _load_credentials()
    root = Path(output_root)
    config = _get_model_config(model)

    return _batch_generate_model(
        sections,
        output_dir=root / config.model,
        resource_id=config.resource_id,
        speaker=config.speaker,
        sample_rate=sample_rate,
        credentials=credentials,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="火山引擎豆包语音合成 HTTP Chunked 最小实现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "PowerShell 环境变量示例:\n"
            '  $env:VOLCENGINE_TTS_APP_ID = "your-app-id"\n'
            '  $env:VOLCENGINE_TTS_ACCESS_KEY = "your-access-key"\n'
            '  $env:VOLCENGINE_TTS_UID = "demo-user"\n\n'
            "示例:\n"
            "  uv run python tts/tts_https.py v1 --text \"您好，欢迎光临。\" --output tts/tts_https_wavs/v1/demo.wav\n"
            "  uv run python tts/tts_https.py v2 --text \"您好，欢迎光临。\" --output tts/tts_https_wavs/v2/demo.wav\n"
            "  uv run python tts/tts_https.py batch --model v1\n"
            "  uv run python tts/tts_https.py batch --model v2\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, default_speaker in (("v1", SPEAKER_V1), ("v2", SPEAKER_V2)):
        subparser = subparsers.add_parser(name, help=f"{name} 单条文本合成")
        subparser.add_argument("--text", required=True, help="待合成文本")
        subparser.add_argument("--output", required=True, help="输出 wav 文件路径")
        subparser.add_argument("--speaker", default=default_speaker, help="音色 ID")
        subparser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="采样率，默认 24000")

    batch_parser = subparsers.add_parser("batch", help="按 Markdown 标题批量导出单个模型 wav")
    batch_parser.add_argument("--model", choices=("v1", "v2"), required=True, help="选择批量导出的模型版本")
    batch_parser.add_argument(
        "--input",
        default=str(DEFAULT_BATCH_INPUT),
        help=f"Markdown 输入文件，默认 {DEFAULT_BATCH_INPUT.name}",
    )
    batch_parser.add_argument("--output-dir", default=str(DEFAULT_BATCH_OUTPUT), help="输出目录，默认 tts_https_wavs")
    batch_parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="采样率，默认 24000")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "v1":
            print(f"[1/1] 开始生成：v1 -> {args.output}", flush=True)
            path = synthesize_tts_v1(args.text, args.output, speaker=args.speaker, sample_rate=args.sample_rate)
            print(f"[1/1] 生成完成：{path}", flush=True)
            return

        if args.command == "v2":
            print(f"[1/1] 开始生成：v2 -> {args.output}", flush=True)
            path = synthesize_tts_v2(args.text, args.output, speaker=args.speaker, sample_rate=args.sample_rate)
            print(f"[1/1] 生成完成：{path}", flush=True)
            return

        paths = synthesize_markdown_cases(
            args.input,
            args.output_dir,
            model=args.model,
            sample_rate=args.sample_rate,
        )
        print(f"{args.model}: generated {len(paths)} new files")
        for path in paths:
            print(path)
    except (ValueError, VolcengineTTSError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
