# -*- coding: utf-8 -*-
"""通用音频处理工具。

`wav` 和 `pcm` 的处理优先使用标准库；`mp3` 相关转换依赖系统中的 `ffmpeg`。
原始 `pcm` 默认按照 16-bit little-endian、单声道、24000 Hz 处理，可按需覆盖。
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import tempfile
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

warnings.filterwarnings("ignore", message="'audioop' is deprecated.*", category=DeprecationWarning)

import audioop

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_MP3_BITRATE = "192k"
DEFAULT_SPLIT_MIN_DURATION = 2.0
DEFAULT_SPLIT_TARGET_DURATION = 5.0
DEFAULT_SPLIT_MAX_DURATION = 7.0
DEFAULT_SPLIT_SILENCE_THRESHOLD_DB = -35.0
DEFAULT_SPLIT_MIN_SILENCE_DURATION = 0.25
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "pcm"}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')
SILENCE_START_PATTERN = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
SILENCE_END_PATTERN = re.compile(
    r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)\s*\|\s*silence_duration:\s*([0-9]+(?:\.[0-9]+)?)"
)
LANGUAGE_TOKEN_MAP = {
    "zh": ("zh", "cn", "chinese"),
    "en": ("en", "english"),
}
LANGUAGE_TEXT_MAP = {
    "zh": ("中文",),
    "en": ("英文",),
}
LOW_ENERGY_WINDOW_SECONDS = 0.05
LOW_ENERGY_STEP_SECONDS = 0.01
WAV_SPLIT_COPY_CHUNK_FRAMES = 32768


class AudioUtilityError(RuntimeError):
    """音频处理失败。"""


@dataclass(frozen=True)
class AudioSpec:
    """描述 PCM/WAV 的基础参数。"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS
    sample_width: int = DEFAULT_SAMPLE_WIDTH

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        if self.channels <= 0:
            raise ValueError("channels 必须大于 0")
        if self.sample_width not in (1, 2, 3, 4):
            raise ValueError("sample_width 仅支持 1/2/3/4 字节")

    @classmethod
    def from_wav(cls, wav_path: str | Path) -> AudioSpec:
        path = _to_path(wav_path)
        _ensure_file_exists(path)
        with wave.open(str(path), "rb") as wav_file:
            return cls(
                sample_rate=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                sample_width=wav_file.getsampwidth(),
            )


@dataclass(frozen=True)
class SplitAudioResult:
    """音频切分结果。"""

    input_path: Path
    language: str
    output_dir: Path
    output_paths: tuple[Path, ...]
    durations: tuple[float, ...]


@dataclass(frozen=True)
class SplitAudioBatchResult:
    """多个音频文件的切分结果。"""

    output_dir: Path
    results: tuple[SplitAudioResult, ...]


def read_wav_metadata(wav_path: str | Path) -> AudioSpec:
    """读取 WAV 文件的音频参数。"""

    return AudioSpec.from_wav(wav_path)


def read_wav_pcm(wav_path: str | Path) -> tuple[bytes, AudioSpec]:
    """读取 WAV 文件，返回 PCM 数据与参数。"""

    path = _to_path(wav_path)
    _ensure_file_exists(path)

    with wave.open(str(path), "rb") as wav_file:
        spec = AudioSpec(
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            sample_width=wav_file.getsampwidth(),
        )
        spec.validate()
        return wav_file.readframes(wav_file.getnframes()), spec


def detect_audio_language(audio_path: str | Path, *, language: str | None = None) -> str:
    """根据路径规则识别音频语言，或返回显式指定的语言。"""

    normalized = _normalize_language(language)
    if normalized != "auto":
        return normalized

    path = _to_path(audio_path)
    ascii_tokens: set[str] = set()
    raw_parts: list[str] = []
    for part in path.parts + (path.stem,):
        normalized_part = part.casefold()
        raw_parts.append(normalized_part)
        ascii_tokens.update(token for token in re.split(r"[^a-z0-9]+", normalized_part) if token)

    matches: set[str] = set()
    for code, tokens in LANGUAGE_TOKEN_MAP.items():
        if any(token in ascii_tokens for token in tokens):
            matches.add(code)
    for code, texts in LANGUAGE_TEXT_MAP.items():
        if any(text in raw_part for raw_part in raw_parts for text in texts):
            matches.add(code)

    if len(matches) == 1:
        return next(iter(matches))
    if matches:
        raise ValueError("无法自动识别语言：路径同时包含中文和英文标记，请显式传入 language='zh' 或 'en'")
    raise ValueError("无法根据音频路径自动识别语言，请显式传入 language='zh' 或 'en'")


def split_wav_on_pauses(
    input_path: str | Path,
    output_dir: str | Path | None,
    split_name: str,
    *,
    min_duration: float = DEFAULT_SPLIT_MIN_DURATION,
    target_duration: float = DEFAULT_SPLIT_TARGET_DURATION,
    max_duration: float = DEFAULT_SPLIT_MAX_DURATION,
    silence_threshold_db: float = DEFAULT_SPLIT_SILENCE_THRESHOLD_DB,
    min_silence_duration: float = DEFAULT_SPLIT_MIN_SILENCE_DURATION,
    language: str | None = None,
    start_index: int | None = None,
) -> SplitAudioResult:
    """按停顿优先切分单个 WAV 音频文件。"""

    _validate_split_durations(
        min_duration=min_duration,
        target_duration=target_duration,
        max_duration=max_duration,
        min_silence_duration=min_silence_duration,
    )

    source = _to_path(input_path)
    _ensure_file_exists(source)
    if _resolve_audio_format(source, None) != "wav":
        raise ValueError("split_wav_on_pauses 仅支持 WAV 输入")

    normalized_language = detect_audio_language(source, language=language)
    normalized_split_name = _sanitize_split_name(split_name)
    target_dir = _resolve_split_output_dir(source, output_dir, normalized_split_name)
    resolved_start_index = _resolve_split_start_index(target_dir, normalized_split_name, start_index)

    with wave.open(str(source), "rb") as wav_file:
        spec = AudioSpec(
            sample_rate=wav_file.getframerate(),
            channels=wav_file.getnchannels(),
            sample_width=wav_file.getsampwidth(),
        )
        spec.validate()
        total_frames = wav_file.getnframes()

    if total_frames <= 0:
        raise AudioUtilityError("输入 WAV 不包含任何音频帧")

    min_frames = _seconds_to_frame_count(min_duration, spec.sample_rate, mode="ceil")
    target_frames = _seconds_to_frame_count(target_duration, spec.sample_rate, mode="round")
    max_frames = _seconds_to_frame_count(max_duration, spec.sample_rate, mode="floor")
    if max_frames < min_frames:
        raise ValueError("max_duration 换算后的帧数必须大于等于 min_duration")
    if total_frames < min_frames:
        raise AudioUtilityError("输入 WAV 总时长小于最小切分时长，无法满足切分约束")

    pause_frames = _detect_pause_midpoint_frames(
        source,
        sample_rate=spec.sample_rate,
        total_frames=total_frames,
        silence_threshold_db=silence_threshold_db,
        min_silence_duration=min_silence_duration,
    )
    frame_boundaries = _choose_split_boundaries(
        source,
        spec,
        total_frames=total_frames,
        pause_frames=pause_frames,
        min_frames=min_frames,
        target_frames=target_frames,
        max_frames=max_frames,
    )
    _validate_split_boundaries(frame_boundaries, min_frames=min_frames, max_frames=max_frames)

    output_paths, durations = _write_wav_segments(
        source,
        target_dir,
        normalized_split_name,
        spec,
        frame_boundaries,
        start_index=resolved_start_index,
    )
    return SplitAudioResult(
        input_path=source,
        language=normalized_language,
        output_dir=target_dir,
        output_paths=tuple(output_paths),
        durations=tuple(durations),
    )


def split_wavs_on_pauses(
    input_paths: Sequence[str | Path],
    output_dir: str | Path | None,
    split_name: str,
    *,
    min_duration: float = DEFAULT_SPLIT_MIN_DURATION,
    target_duration: float = DEFAULT_SPLIT_TARGET_DURATION,
    max_duration: float = DEFAULT_SPLIT_MAX_DURATION,
    silence_threshold_db: float = DEFAULT_SPLIT_SILENCE_THRESHOLD_DB,
    min_silence_duration: float = DEFAULT_SPLIT_MIN_SILENCE_DURATION,
    language: str | None = None,
) -> SplitAudioBatchResult:
    """按顺序切分多个 WAV 文件，并使用连续编号输出。"""

    if not input_paths:
        raise ValueError("input_paths 不能为空")

    normalized_paths = [_to_path(path) for path in input_paths]
    for source in normalized_paths:
        _ensure_file_exists(source)
        if _resolve_audio_format(source, None) != "wav":
            raise ValueError("split_wavs_on_pauses 仅支持 WAV 输入")

    normalized_split_name = _sanitize_split_name(split_name)
    target_dir = _resolve_batch_split_output_dir(normalized_paths, output_dir, normalized_split_name)
    current_index = _find_next_split_index(target_dir, normalized_split_name)

    results: list[SplitAudioResult] = []
    for source in normalized_paths:
        result = split_wav_on_pauses(
            source,
            target_dir,
            normalized_split_name,
            min_duration=min_duration,
            target_duration=target_duration,
            max_duration=max_duration,
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_silence_duration,
            language=language,
            start_index=current_index,
        )
        results.append(result)
        current_index += len(result.output_paths)

    return SplitAudioBatchResult(output_dir=target_dir, results=tuple(results))


def write_wav(
    pcm_data: bytes,
    wav_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """将原始 PCM 数据写入 WAV 文件。"""

    spec = AudioSpec(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
    spec.validate()
    _validate_pcm_length(pcm_data, spec.sample_width, spec.channels)

    target = _to_path(wav_path)
    _ensure_parent_dir(target)

    with wave.open(str(target), "wb") as wav_file:
        wav_file.setnchannels(spec.channels)
        wav_file.setsampwidth(spec.sample_width)
        wav_file.setframerate(spec.sample_rate)
        wav_file.writeframes(pcm_data)

    return target


def convert_wav_to_pcm(wav_path: str | Path, pcm_path: str | Path) -> Path:
    """将 WAV 文件提取为原始 PCM 数据。"""

    pcm_data, _ = read_wav_pcm(wav_path)
    target = _to_path(pcm_path)
    _ensure_parent_dir(target)
    target.write_bytes(pcm_data)
    return target


def convert_pcm_to_wav(
    pcm_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """将原始 PCM 数据封装为 WAV 文件。"""

    source = _to_path(pcm_path)
    _ensure_file_exists(source)
    pcm_data = source.read_bytes()
    return write_wav(
        pcm_data,
        wav_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def convert_wav_to_mp3(
    wav_path: str | Path,
    mp3_path: str | Path,
    *,
    bitrate: str = DEFAULT_MP3_BITRATE,
) -> Path:
    """将 WAV 转为 MP3。"""

    source = _to_path(wav_path)
    _ensure_file_exists(source)
    target = _to_path(mp3_path)
    _ensure_parent_dir(target)

    _run_ffmpeg(
        [
            "-i",
            str(source),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(target),
        ]
    )
    return target


def convert_mp3_to_wav(
    mp3_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """将 MP3 转为 WAV。"""

    return _decode_mp3_to_wav(
        mp3_path,
        wav_path,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
    )


def convert_pcm_to_mp3(
    pcm_path: str | Path,
    mp3_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
    bitrate: str = DEFAULT_MP3_BITRATE,
) -> Path:
    """将原始 PCM 转为 MP3。"""

    source = _to_path(pcm_path)
    _ensure_file_exists(source)
    spec = AudioSpec(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
    spec.validate()

    target = _to_path(mp3_path)
    _ensure_parent_dir(target)

    _run_ffmpeg(
        [
            "-f",
            _pcm_format_name(spec.sample_width),
            "-ar",
            str(spec.sample_rate),
            "-ac",
            str(spec.channels),
            "-i",
            str(source),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(target),
        ]
    )
    return target


def convert_mp3_to_pcm(
    mp3_path: str | Path,
    pcm_path: str | Path,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """将 MP3 转为原始 PCM。"""

    source = _to_path(mp3_path)
    _ensure_file_exists(source)
    spec = AudioSpec(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
    spec.validate()

    target = _to_path(pcm_path)
    _ensure_parent_dir(target)

    _run_ffmpeg(
        [
            "-i",
            str(source),
            "-vn",
            "-ar",
            str(spec.sample_rate),
            "-ac",
            str(spec.channels),
            "-f",
            _pcm_format_name(spec.sample_width),
            "-acodec",
            _pcm_codec_name(spec.sample_width),
            str(target),
        ]
    )
    return target


def convert_audio_format(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_format: str | None = None,
    output_format: str | None = None,
    sample_rate: int | None = None,
    channels: int | None = None,
    sample_width: int | None = None,
    bitrate: str = DEFAULT_MP3_BITRATE,
) -> Path:
    """通用音频格式转换入口。

    说明：
    - 当输入或输出为 `pcm` 时，`sample_rate/channels/sample_width` 描述的是该原始 PCM 的格式。
    - 当输出为 `wav` 且来源是 `mp3` 时，这些参数描述输出 WAV 的目标格式。
    """

    source = _to_path(input_path)
    target = _to_path(output_path)
    _ensure_file_exists(source)
    source_format = _resolve_audio_format(source, input_format)
    target_format = _resolve_audio_format(target, output_format)

    if source_format == target_format:
        if source_format == "wav":
            source_spec = read_wav_metadata(source)
            return resample_wav(
                source,
                target,
                sample_rate=sample_rate or source_spec.sample_rate,
                channels=channels or source_spec.channels,
                sample_width=sample_width or source_spec.sample_width,
            )
        if source_format == "pcm":
            if sample_rate is not None or channels is not None or sample_width is not None:
                spec = AudioSpec(
                    sample_rate=sample_rate or DEFAULT_SAMPLE_RATE,
                    channels=channels or DEFAULT_CHANNELS,
                    sample_width=sample_width or DEFAULT_SAMPLE_WIDTH,
                )
                spec.validate()
            _ensure_parent_dir(target)
            target.write_bytes(source.read_bytes())
            return target
        command = ["-i", str(source), "-vn"]
        if sample_rate:
            command.extend(["-ar", str(sample_rate)])
        if channels:
            command.extend(["-ac", str(channels)])
        command.extend(["-codec:a", "libmp3lame", "-b:a", bitrate, str(target)])
        _ensure_parent_dir(target)
        _run_ffmpeg(command)
        return target

    effective_sample_rate = sample_rate or DEFAULT_SAMPLE_RATE
    effective_channels = channels or DEFAULT_CHANNELS
    effective_sample_width = sample_width or DEFAULT_SAMPLE_WIDTH

    if source_format == "wav" and target_format == "pcm":
        return convert_wav_to_pcm(source, target)
    if source_format == "pcm" and target_format == "wav":
        return convert_pcm_to_wav(
            source,
            target,
            sample_rate=effective_sample_rate,
            channels=effective_channels,
            sample_width=effective_sample_width,
        )
    if source_format == "wav" and target_format == "mp3":
        return convert_wav_to_mp3(source, target, bitrate=bitrate)
    if source_format == "mp3" and target_format == "wav":
        return convert_mp3_to_wav(
            source,
            target,
            sample_rate=effective_sample_rate,
            channels=effective_channels,
            sample_width=effective_sample_width,
        )
    if source_format == "pcm" and target_format == "mp3":
        return convert_pcm_to_mp3(
            source,
            target,
            sample_rate=effective_sample_rate,
            channels=effective_channels,
            sample_width=effective_sample_width,
            bitrate=bitrate,
        )
    if source_format == "mp3" and target_format == "pcm":
        return convert_mp3_to_pcm(
            source,
            target,
            sample_rate=effective_sample_rate,
            channels=effective_channels,
            sample_width=effective_sample_width,
        )

    raise AudioUtilityError(f"暂不支持从 {source_format} 转换到 {target_format}")


def concatenate_wav_files(
    wav_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    sample_width: int | None = None,
) -> Path:
    """拼接多个 WAV 文件。

    如果传入了目标采样率、声道数或位宽，会在拼接前自动统一到目标格式。
    未传入时默认沿用第一个 WAV 的参数。
    """

    if not wav_paths:
        raise ValueError("wav_paths 不能为空")

    normalized_paths = [_to_path(path) for path in wav_paths]
    for path in normalized_paths:
        _ensure_file_exists(path)

    _, first_spec = read_wav_pcm(normalized_paths[0])
    target_spec = AudioSpec(
        sample_rate=sample_rate or first_spec.sample_rate,
        channels=channels or first_spec.channels,
        sample_width=sample_width or first_spec.sample_width,
    )
    target_spec.validate()

    merged = bytearray()
    for path in normalized_paths:
        pcm_data, source_spec = read_wav_pcm(path)
        merged.extend(_transform_pcm(pcm_data, source_spec, target_spec))

    return write_wav(
        bytes(merged),
        output_path,
        sample_rate=target_spec.sample_rate,
        channels=target_spec.channels,
        sample_width=target_spec.sample_width,
    )


def resample_wav(
    wav_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int,
    channels: int | None = None,
    sample_width: int | None = None,
) -> Path:
    """重采样 WAV，并可选调整声道数和采样位宽。"""

    pcm_data, source_spec = read_wav_pcm(wav_path)
    target_spec = AudioSpec(
        sample_rate=sample_rate,
        channels=channels or source_spec.channels,
        sample_width=sample_width or source_spec.sample_width,
    )
    target_spec.validate()

    converted = _transform_pcm(pcm_data, source_spec, target_spec)
    return write_wav(
        converted,
        output_path,
        sample_rate=target_spec.sample_rate,
        channels=target_spec.channels,
        sample_width=target_spec.sample_width,
    )


def resample_audio_directory(
    input_dir: str | Path,
    *,
    sample_rate: int,
    pcm_sample_rate: int = DEFAULT_SAMPLE_RATE,
    pcm_channels: int = DEFAULT_CHANNELS,
    pcm_sample_width: int = DEFAULT_SAMPLE_WIDTH,
    mp3_bitrate: str = DEFAULT_MP3_BITRATE,
) -> list[Path]:
    """原地重采样目录下的所有受支持音频。"""

    audio_paths = _collect_audio_files(input_dir)
    pcm_spec = AudioSpec(
        sample_rate=pcm_sample_rate,
        channels=pcm_channels,
        sample_width=pcm_sample_width,
    )
    pcm_spec.validate()

    processed_paths: list[Path] = []
    for audio_path in audio_paths:
        audio_format = _resolve_audio_format(audio_path, None)
        temp_output_path = _create_temp_output_path(audio_path.parent, audio_path.suffix)
        try:
            if audio_format == "wav":
                source_spec = read_wav_metadata(audio_path)
                resample_wav(
                    audio_path,
                    temp_output_path,
                    sample_rate=sample_rate,
                    channels=source_spec.channels,
                    sample_width=source_spec.sample_width,
                )
            elif audio_format == "mp3":
                with tempfile.TemporaryDirectory(prefix="audio-resample-", dir=str(audio_path.parent)) as temp_dir:
                    temp_root = Path(temp_dir)
                    decoded_path = temp_root / f"{audio_path.stem}.decoded.wav"
                    resampled_path = temp_root / f"{audio_path.stem}.resampled.wav"
                    _decode_mp3_to_wav(audio_path, decoded_path, sample_rate=None, channels=None, sample_width=None)
                    decoded_spec = read_wav_metadata(decoded_path)
                    resample_wav(
                        decoded_path,
                        resampled_path,
                        sample_rate=sample_rate,
                        channels=decoded_spec.channels,
                        sample_width=decoded_spec.sample_width,
                    )
                    convert_wav_to_mp3(resampled_path, temp_output_path, bitrate=mp3_bitrate)
            else:
                pcm_data = audio_path.read_bytes()
                _validate_pcm_length(pcm_data, pcm_spec.sample_width, pcm_spec.channels)
                target_spec = AudioSpec(
                    sample_rate=sample_rate,
                    channels=pcm_spec.channels,
                    sample_width=pcm_spec.sample_width,
                )
                converted = _transform_pcm(pcm_data, pcm_spec, target_spec)
                temp_output_path.write_bytes(converted)

            temp_output_path.replace(audio_path)
            processed_paths.append(audio_path)
        except Exception:
            if temp_output_path.exists():
                temp_output_path.unlink()
            raise

    return processed_paths


def concatenate_audio_directory(
    input_dir: str | Path,
    *,
    output_name: str = "total.wav",
    pcm_sample_rate: int = DEFAULT_SAMPLE_RATE,
    pcm_channels: int = DEFAULT_CHANNELS,
    pcm_sample_width: int = DEFAULT_SAMPLE_WIDTH,
) -> Path:
    """按排序顺序拼接目录下的所有受支持音频为一个 WAV。"""

    output_filename = Path(output_name).name
    if not output_filename.lower().endswith(".wav"):
        raise ValueError("output_name 必须使用 .wav 扩展名")

    audio_paths = _collect_audio_files(input_dir, exclude_names={output_filename.lower()})
    output_dir = _to_path(input_dir)
    output_path = output_dir / output_filename

    with tempfile.TemporaryDirectory(prefix="audio-splice-", dir=str(output_dir)) as temp_dir:
        temp_root = Path(temp_dir)
        first_pcm, first_spec = _read_audio_as_pcm(
            audio_paths[0],
            temp_root,
            pcm_sample_rate=pcm_sample_rate,
            pcm_channels=pcm_channels,
            pcm_sample_width=pcm_sample_width,
        )
        merged = bytearray(first_pcm)

        for audio_path in audio_paths[1:]:
            pcm_data, source_spec = _read_audio_as_pcm(
                audio_path,
                temp_root,
                pcm_sample_rate=pcm_sample_rate,
                pcm_channels=pcm_channels,
                pcm_sample_width=pcm_sample_width,
            )
            merged.extend(_transform_pcm(pcm_data, source_spec, first_spec))

    return write_wav(
        bytes(merged),
        output_path,
        sample_rate=first_spec.sample_rate,
        channels=first_spec.channels,
        sample_width=first_spec.sample_width,
    )


def _validate_split_durations(
    *,
    min_duration: float,
    target_duration: float,
    max_duration: float,
    min_silence_duration: float,
) -> None:
    if min_duration <= 0:
        raise ValueError("min_duration 必须大于 0")
    if target_duration < min_duration:
        raise ValueError("target_duration 必须大于等于 min_duration")
    if max_duration < target_duration:
        raise ValueError("max_duration 必须大于等于 target_duration")
    if min_silence_duration <= 0:
        raise ValueError("min_silence_duration 必须大于 0")


def _normalize_language(language: str | None) -> str:
    normalized = (language or "auto").strip().casefold()
    mapping = {
        "auto": "auto",
        "zh": "zh",
        "cn": "zh",
        "chinese": "zh",
        "中文": "zh",
        "en": "en",
        "english": "en",
        "英文": "en",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError("language 仅支持 auto、zh、cn、中文、en、english、英文") from exc


def _sanitize_split_name(split_name: str) -> str:
    sanitized = INVALID_FILENAME_CHARS.sub("_", split_name).strip().rstrip(". ")
    if not sanitized:
        raise ValueError("split_name 不能为空")
    return sanitized


def _resolve_split_output_dir(input_path: Path, output_dir: str | Path | None, split_name: str) -> Path:
    if output_dir is None:
        target_dir = input_path.parent / split_name
    else:
        target_dir = _to_path(output_dir)

    if target_dir.exists() and not target_dir.is_dir():
        raise NotADirectoryError(f"输出目录不是有效目录：{target_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _resolve_batch_split_output_dir(input_paths: Sequence[Path], output_dir: str | Path | None, split_name: str) -> Path:
    if not input_paths:
        raise ValueError("input_paths 不能为空")
    return _resolve_split_output_dir(input_paths[0], output_dir, split_name)


def _resolve_split_start_index(output_dir: Path, split_name: str, start_index: int | None) -> int:
    if start_index is None:
        return _find_next_split_index(output_dir, split_name)
    if start_index < 0:
        raise ValueError("start_index 不能小于 0")
    return start_index


def _find_next_split_index(output_dir: Path, split_name: str) -> int:
    if not output_dir.exists():
        return 0

    pattern = re.compile(rf"^{re.escape(split_name)}_(\d+)\.wav$")
    max_index = -1
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _seconds_to_frame_count(seconds: float, sample_rate: int, *, mode: str) -> int:
    frames = seconds * sample_rate
    if mode == "ceil":
        return max(1, math.ceil(frames))
    if mode == "floor":
        return max(1, math.floor(frames))
    if mode == "round":
        return max(1, round(frames))
    raise ValueError(f"未知的帧数换算模式：{mode}")


def _detect_pause_midpoint_frames(
    wav_path: Path,
    *,
    sample_rate: int,
    total_frames: int,
    silence_threshold_db: float,
    min_silence_duration: float,
) -> list[int]:
    output = _run_ffmpeg_command(
        [
            "-i",
            str(wav_path),
            "-af",
            f"silencedetect=n={silence_threshold_db}dB:d={min_silence_duration}",
            "-f",
            "null",
            "-",
        ],
        loglevel="info",
    )
    raw_output = "\n".join(part for part in (output.stdout, output.stderr) if part)

    pause_frames: list[int] = []
    current_start: float | None = None
    upper_bound = max(1, total_frames - 1)
    for line in raw_output.splitlines():
        start_match = SILENCE_START_PATTERN.search(line)
        if start_match:
            current_start = float(start_match.group(1))
            continue

        end_match = SILENCE_END_PATTERN.search(line)
        if not end_match:
            continue

        silence_end = float(end_match.group(1))
        silence_duration = float(end_match.group(2))
        silence_start = current_start if current_start is not None else max(0.0, silence_end - silence_duration)
        current_start = None

        midpoint_seconds = max(0.0, (silence_start + silence_end) / 2.0)
        frame = min(upper_bound, max(1, _seconds_to_frame_count(midpoint_seconds, sample_rate, mode="round")))
        pause_frames.append(frame)

    return sorted(set(pause_frames))


def _choose_split_boundaries(
    wav_path: Path,
    spec: AudioSpec,
    *,
    total_frames: int,
    pause_frames: Sequence[int],
    min_frames: int,
    target_frames: int,
    max_frames: int,
) -> list[int]:
    boundaries = [0]
    current_frame = 0

    while total_frames - current_frame > max_frames:
        min_split_frame = current_frame + min_frames
        max_split_frame = min(current_frame + max_frames, total_frames - min_frames)
        preferred_frame = min(max(current_frame + target_frames, min_split_frame), max_split_frame)

        candidate_frames = [frame for frame in pause_frames if min_split_frame <= frame <= max_split_frame]
        if candidate_frames:
            split_frame = min(candidate_frames, key=lambda frame: (abs(frame - preferred_frame), frame))
        else:
            split_frame = _find_low_energy_split_frame(
                wav_path,
                spec,
                lower_frame=min_split_frame,
                upper_frame=max_split_frame,
                preferred_frame=preferred_frame,
            )

        if split_frame <= current_frame or split_frame >= total_frames:
            raise AudioUtilityError("无法为当前音频计算有效的切分边界")

        boundaries.append(split_frame)
        current_frame = split_frame

    boundaries.append(total_frames)
    return boundaries


def _find_low_energy_split_frame(
    wav_path: Path,
    spec: AudioSpec,
    *,
    lower_frame: int,
    upper_frame: int,
    preferred_frame: int,
) -> int:
    preferred_frame = min(max(preferred_frame, lower_frame), upper_frame)
    if upper_frame <= lower_frame:
        return preferred_frame

    window_frames = min(
        upper_frame - lower_frame + 1,
        _seconds_to_frame_count(LOW_ENERGY_WINDOW_SECONDS, spec.sample_rate, mode="round"),
    )
    step_frames = max(1, _seconds_to_frame_count(LOW_ENERGY_STEP_SECONDS, spec.sample_rate, mode="round"))
    half_window = max(1, window_frames // 2)

    with wave.open(str(wav_path), "rb") as wav_file:
        total_frames = wav_file.getnframes()
        region_start = max(0, lower_frame - half_window)
        region_end = min(total_frames, upper_frame + half_window)
        wav_file.setpos(region_start)
        region_pcm = wav_file.readframes(region_end - region_start)

    frame_width = spec.sample_width * spec.channels
    if not region_pcm:
        return preferred_frame

    best_score: tuple[int, int] | None = None
    best_frame = preferred_frame
    for candidate_frame in range(lower_frame, upper_frame + 1, step_frames):
        window_start = max(region_start, candidate_frame - half_window)
        window_end = min(region_end, candidate_frame + half_window)
        start_index = (window_start - region_start) * frame_width
        end_index = (window_end - region_start) * frame_width
        window_pcm = region_pcm[start_index:end_index]
        if not window_pcm:
            continue

        rms = audioop.rms(window_pcm, spec.sample_width)
        score = (rms, abs(candidate_frame - preferred_frame))
        if best_score is None or score < best_score:
            best_score = score
            best_frame = candidate_frame

    return best_frame


def _validate_split_boundaries(boundaries: Sequence[int], *, min_frames: int, max_frames: int) -> None:
    if len(boundaries) < 2:
        raise AudioUtilityError("切分边界数量不足")

    previous = boundaries[0]
    for current in boundaries[1:]:
        segment_frames = current - previous
        if segment_frames < min_frames:
            raise AudioUtilityError("切分后存在小于最小时长的音频片段")
        if segment_frames > max_frames:
            raise AudioUtilityError("切分后存在大于最大时长的音频片段")
        previous = current


def _write_wav_segments(
    input_path: Path,
    output_dir: Path,
    split_name: str,
    spec: AudioSpec,
    frame_boundaries: Sequence[int],
    *,
    start_index: int,
) -> tuple[list[Path], list[float]]:
    output_paths: list[Path] = []
    durations: list[float] = []

    with wave.open(str(input_path), "rb") as source_wav:
        current_position = 0
        frame_width = spec.sample_width * spec.channels

        for index, (start_frame, end_frame) in enumerate(zip(frame_boundaries, frame_boundaries[1:])):
            if current_position != start_frame:
                source_wav.setpos(start_frame)
                current_position = start_frame

            output_path = output_dir / f"{split_name}_{start_index + index}.wav"
            if output_path.exists():
                raise AudioUtilityError(f"切分输出文件已存在：{output_path}")
            with wave.open(str(output_path), "wb") as target_wav:
                target_wav.setnchannels(spec.channels)
                target_wav.setsampwidth(spec.sample_width)
                target_wav.setframerate(spec.sample_rate)

                remaining_frames = end_frame - start_frame
                while remaining_frames > 0:
                    chunk_frames = min(remaining_frames, WAV_SPLIT_COPY_CHUNK_FRAMES)
                    chunk_pcm = source_wav.readframes(chunk_frames)
                    if not chunk_pcm:
                        raise AudioUtilityError("切分 WAV 时遇到意外的文件结尾")

                    actual_frames = len(chunk_pcm) // frame_width
                    if actual_frames <= 0:
                        raise AudioUtilityError("切分 WAV 时读取到无效的 PCM 数据块")

                    target_wav.writeframes(chunk_pcm)
                    remaining_frames -= actual_frames
                    current_position += actual_frames

            output_paths.append(output_path)
            durations.append((end_frame - start_frame) / spec.sample_rate)

    return output_paths, durations


def _transform_pcm(pcm_data: bytes, source_spec: AudioSpec, target_spec: AudioSpec) -> bytes:
    """统一 PCM 的采样率、声道数和位宽。"""

    source_spec.validate()
    target_spec.validate()
    _validate_pcm_length(pcm_data, source_spec.sample_width, source_spec.channels)

    converted = pcm_data
    current_width = source_spec.sample_width
    current_channels = source_spec.channels
    current_rate = source_spec.sample_rate

    if current_channels != target_spec.channels:
        converted = _convert_channels(converted, current_width, current_channels, target_spec.channels)
        current_channels = target_spec.channels

    if current_rate != target_spec.sample_rate:
        converted, _ = audioop.ratecv(
            converted,
            current_width,
            current_channels,
            current_rate,
            target_spec.sample_rate,
            None,
        )
        current_rate = target_spec.sample_rate

    if current_width != target_spec.sample_width:
        converted = audioop.lin2lin(converted, current_width, target_spec.sample_width)

    _validate_pcm_length(converted, target_spec.sample_width, target_spec.channels)
    return converted


def _convert_channels(pcm_data: bytes, sample_width: int, source_channels: int, target_channels: int) -> bytes:
    if source_channels == target_channels:
        return pcm_data

    if source_channels == 1 and target_channels == 2:
        return audioop.tostereo(pcm_data, sample_width, 1.0, 1.0)

    if source_channels == 2 and target_channels == 1:
        return audioop.tomono(pcm_data, sample_width, 0.5, 0.5)

    raise AudioUtilityError("当前仅支持单声道和双声道之间互转")


def _pcm_format_name(sample_width: int) -> str:
    mapping = {
        1: "u8",
        2: "s16le",
        3: "s24le",
        4: "s32le",
    }
    try:
        return mapping[sample_width]
    except KeyError as exc:
        raise ValueError("sample_width 仅支持 1/2/3/4 字节") from exc


def _pcm_codec_name(sample_width: int) -> str:
    mapping = {
        1: "pcm_u8",
        2: "pcm_s16le",
        3: "pcm_s24le",
        4: "pcm_s32le",
    }
    try:
        return mapping[sample_width]
    except KeyError as exc:
        raise ValueError("sample_width 仅支持 1/2/3/4 字节") from exc


def _resolve_audio_format(path: Path, explicit_format: str | None) -> str:
    if explicit_format:
        normalized = explicit_format.strip().lower().lstrip(".")
    else:
        normalized = path.suffix.lower().lstrip(".")

    if normalized not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(f"不支持的音频格式：{normalized or '<empty>'}")

    return normalized


def _collect_audio_files(input_dir: str | Path, *, exclude_names: set[str] | None = None) -> list[Path]:
    directory = _to_path(input_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"目录不存在：{directory}")

    excluded = {name.lower() for name in (exclude_names or set())}
    audio_paths = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower().lstrip(".") in SUPPORTED_AUDIO_FORMATS
        and path.name.lower() not in excluded
    ]
    audio_paths.sort(key=lambda path: path.name.lower())

    if not audio_paths:
        raise AudioUtilityError(f"目录中未找到可处理的音频文件：{directory}")

    return audio_paths


def _create_temp_output_path(directory: Path, suffix: str) -> Path:
    with tempfile.NamedTemporaryFile(prefix=".audio-", suffix=suffix, dir=directory, delete=False) as handle:
        return Path(handle.name)


def _read_audio_as_pcm(
    audio_path: Path,
    temp_dir: Path,
    *,
    pcm_sample_rate: int,
    pcm_channels: int,
    pcm_sample_width: int,
) -> tuple[bytes, AudioSpec]:
    audio_format = _resolve_audio_format(audio_path, None)
    if audio_format == "wav":
        return read_wav_pcm(audio_path)
    if audio_format == "pcm":
        spec = AudioSpec(
            sample_rate=pcm_sample_rate,
            channels=pcm_channels,
            sample_width=pcm_sample_width,
        )
        spec.validate()
        pcm_data = audio_path.read_bytes()
        _validate_pcm_length(pcm_data, spec.sample_width, spec.channels)
        return pcm_data, spec

    decoded_path = _create_temp_output_path(temp_dir, ".wav")
    try:
        _decode_mp3_to_wav(audio_path, decoded_path, sample_rate=None, channels=None, sample_width=None)
        return read_wav_pcm(decoded_path)
    finally:
        if decoded_path.exists():
            decoded_path.unlink()


def _decode_mp3_to_wav(
    mp3_path: str | Path,
    wav_path: str | Path,
    *,
    sample_rate: int | None,
    channels: int | None,
    sample_width: int | None,
) -> Path:
    source = _to_path(mp3_path)
    _ensure_file_exists(source)
    target = _to_path(wav_path)
    _ensure_parent_dir(target)

    command = ["-i", str(source), "-vn"]
    if sample_rate is not None:
        if sample_rate <= 0:
            raise ValueError("sample_rate 必须大于 0")
        command.extend(["-ar", str(sample_rate)])
    if channels is not None:
        if channels <= 0:
            raise ValueError("channels 必须大于 0")
        command.extend(["-ac", str(channels)])
    if sample_width is not None:
        if sample_width not in (1, 2, 3, 4):
            raise ValueError("sample_width 仅支持 1/2/3/4 字节")
        command.extend(["-acodec", _pcm_codec_name(sample_width)])
    command.append(str(target))

    _run_ffmpeg(command)
    return target


def _run_ffmpeg(args: Sequence[str]) -> None:
    _run_ffmpeg_command(args, loglevel="error")


def _run_ffmpeg_command(args: Sequence[str], *, loglevel: str) -> subprocess.CompletedProcess[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioUtilityError("处理 mp3 需要先安装 ffmpeg，并确保 ffmpeg 已加入 PATH")

    command = [ffmpeg, "-y", "-hide_banner", "-nostats", "-loglevel", loglevel, *args]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "未知 ffmpeg 错误"
        raise AudioUtilityError(f"ffmpeg 执行失败：{message}")
    return completed


def _validate_pcm_length(pcm_data: bytes, sample_width: int, channels: int) -> None:
    frame_width = sample_width * channels
    if frame_width <= 0:
        raise ValueError("无效的 frame_width")
    if len(pcm_data) % frame_width != 0:
        raise AudioUtilityError("PCM 数据长度与 sample_width/channels 不匹配")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_file_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")


def _to_path(path: str | Path) -> Path:
    return Path(path)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通用音频处理工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    resample_parser = subparsers.add_parser("resample-dir", help="原地重采样目录下的所有音频")
    resample_parser.add_argument("--input-dir", required=True, help="音频目录路径")
    resample_parser.add_argument("--sample-rate", type=int, required=True, help="目标采样率，例如 48000")
    resample_parser.add_argument(
        "--pcm-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"原始 PCM 的输入采样率，默认 {DEFAULT_SAMPLE_RATE}",
    )
    resample_parser.add_argument(
        "--pcm-channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"原始 PCM 的输入声道数，默认 {DEFAULT_CHANNELS}",
    )
    resample_parser.add_argument(
        "--pcm-sample-width",
        type=int,
        default=DEFAULT_SAMPLE_WIDTH,
        help=f"原始 PCM 的输入采样位宽字节数，默认 {DEFAULT_SAMPLE_WIDTH}",
    )
    resample_parser.add_argument(
        "--bitrate",
        default=DEFAULT_MP3_BITRATE,
        help=f"MP3 重编码码率，默认 {DEFAULT_MP3_BITRATE}",
    )

    splice_parser = subparsers.add_parser("splice-dir", help="按顺序拼接目录下的所有音频为一个 WAV")
    splice_parser.add_argument("--input-dir", required=True, help="音频目录路径")
    splice_parser.add_argument("--output-name", default="total.wav", help="输出文件名，默认 total.wav")
    splice_parser.add_argument(
        "--pcm-sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"原始 PCM 的输入采样率，默认 {DEFAULT_SAMPLE_RATE}",
    )
    splice_parser.add_argument(
        "--pcm-channels",
        type=int,
        default=DEFAULT_CHANNELS,
        help=f"原始 PCM 的输入声道数，默认 {DEFAULT_CHANNELS}",
    )
    splice_parser.add_argument(
        "--pcm-sample-width",
        type=int,
        default=DEFAULT_SAMPLE_WIDTH,
        help=f"原始 PCM 的输入采样位宽字节数，默认 {DEFAULT_SAMPLE_WIDTH}",
    )

    split_parser = subparsers.add_parser("split-wav", help="按停顿切分一个或多个 WAV 文件")
    split_parser.add_argument("--input-file", required=True, action="append", help="输入 WAV 文件路径，可重复传入")
    split_parser.add_argument("--split-name", required=True, help="切分后文件名前缀")
    split_parser.add_argument("--output-dir", help="输出目录，默认使用输入文件同级的 <split-name> 子目录")
    split_parser.add_argument("--language", default="auto", help="语言覆盖：auto / zh / en")
    split_parser.add_argument("--min-duration", type=float, default=DEFAULT_SPLIT_MIN_DURATION, help="最小时长（秒）")
    split_parser.add_argument("--target-duration", type=float, default=DEFAULT_SPLIT_TARGET_DURATION, help="目标时长（秒）")
    split_parser.add_argument("--max-duration", type=float, default=DEFAULT_SPLIT_MAX_DURATION, help="最大时长（秒）")
    split_parser.add_argument(
        "--silence-threshold-db",
        type=float,
        default=DEFAULT_SPLIT_SILENCE_THRESHOLD_DB,
        help="ffmpeg silencedetect 的静音阈值（dB）",
    )
    split_parser.add_argument(
        "--min-silence-duration",
        type=float,
        default=DEFAULT_SPLIT_MIN_SILENCE_DURATION,
        help="ffmpeg silencedetect 的最小静音时长（秒）",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.command == "resample-dir":
            processed_paths = resample_audio_directory(
                args.input_dir,
                sample_rate=args.sample_rate,
                pcm_sample_rate=args.pcm_sample_rate,
                pcm_channels=args.pcm_channels,
                pcm_sample_width=args.pcm_sample_width,
                mp3_bitrate=args.bitrate,
            )
            print(f"扫描到 {len(processed_paths)} 个音频文件")
            print("实际处理顺序：")
            for path in processed_paths:
                print(f"- {path.name}")
            print(f"输出路径：{_to_path(args.input_dir)}")
            return 0

        if args.command == "split-wav":
            result = split_wavs_on_pauses(
                args.input_file,
                args.output_dir,
                args.split_name,
                min_duration=args.min_duration,
                target_duration=args.target_duration,
                max_duration=args.max_duration,
                silence_threshold_db=args.silence_threshold_db,
                min_silence_duration=args.min_silence_duration,
                language=args.language,
            )
            print(f"输出目录：{result.output_dir}")
            print(f"输入文件数：{len(result.results)}")
            print(f"总生成文件数：{sum(len(item.output_paths) for item in result.results)}")
            for item in result.results:
                print(f"输入文件：{item.input_path}")
                print(f"语言类型：{item.language}")
                print(f"生成文件数：{len(item.output_paths)}")
                print("切分文件：")
                for path in item.output_paths:
                    print(f"- {path.name}")
            return 0

        output_path = concatenate_audio_directory(
            args.input_dir,
            output_name=args.output_name,
            pcm_sample_rate=args.pcm_sample_rate,
            pcm_channels=args.pcm_channels,
            pcm_sample_width=args.pcm_sample_width,
        )
        input_paths = _collect_audio_files(args.input_dir, exclude_names={Path(args.output_name).name.lower()})
        metadata = read_wav_metadata(output_path)
        print(f"扫描到 {len(input_paths)} 个音频文件")
        print("实际处理顺序：")
        for path in input_paths:
            print(f"- {path.name}")
        print(f"输出路径：{output_path}")
        print(
            "输出规格："
            f"sample_rate={metadata.sample_rate}, "
            f"channels={metadata.channels}, "
            f"sample_width={metadata.sample_width}"
        )
        return 0
    except (AudioUtilityError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AudioSpec",
    "AudioUtilityError",
    "DEFAULT_CHANNELS",
    "DEFAULT_MP3_BITRATE",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_WIDTH",
    "DEFAULT_SPLIT_MAX_DURATION",
    "DEFAULT_SPLIT_MIN_DURATION",
    "DEFAULT_SPLIT_MIN_SILENCE_DURATION",
    "DEFAULT_SPLIT_SILENCE_THRESHOLD_DB",
    "DEFAULT_SPLIT_TARGET_DURATION",
    "SUPPORTED_AUDIO_FORMATS",
    "SplitAudioBatchResult",
    "SplitAudioResult",
    "concatenate_audio_directory",
    "concatenate_wav_files",
    "convert_audio_format",
    "convert_mp3_to_pcm",
    "convert_mp3_to_wav",
    "convert_pcm_to_mp3",
    "convert_pcm_to_wav",
    "convert_wav_to_mp3",
    "convert_wav_to_pcm",
    "detect_audio_language",
    "read_wav_metadata",
    "read_wav_pcm",
    "resample_audio_directory",
    "resample_wav",
    "split_wav_on_pauses",
    "split_wavs_on_pauses",
    "write_wav",
]
