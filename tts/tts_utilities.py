# -*- coding: utf-8 -*-
"""通用音频处理工具。

`wav` 和 `pcm` 的处理优先使用标准库；`mp3` 相关转换依赖系统中的 `ffmpeg`。
原始 `pcm` 默认按照 16-bit little-endian、单声道、24000 Hz 处理，可按需覆盖。
"""

from __future__ import annotations

import audioop
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_MP3_BITRATE = "192k"
SUPPORTED_AUDIO_FORMATS = {"wav", "mp3", "pcm"}


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

    source = _to_path(mp3_path)
    _ensure_file_exists(source)
    spec = AudioSpec(sample_rate=sample_rate, channels=channels, sample_width=sample_width)
    spec.validate()

    target = _to_path(wav_path)
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
            "-acodec",
            _pcm_codec_name(spec.sample_width),
            str(target),
        ]
    )
    return target


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


def _run_ffmpeg(args: Sequence[str]) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AudioUtilityError("处理 mp3 需要先安装 ffmpeg，并确保 ffmpeg 已加入 PATH")

    command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *args]
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


__all__ = [
    "AudioSpec",
    "AudioUtilityError",
    "DEFAULT_CHANNELS",
    "DEFAULT_MP3_BITRATE",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SAMPLE_WIDTH",
    "SUPPORTED_AUDIO_FORMATS",
    "concatenate_wav_files",
    "convert_audio_format",
    "convert_mp3_to_pcm",
    "convert_mp3_to_wav",
    "convert_pcm_to_mp3",
    "convert_pcm_to_wav",
    "convert_wav_to_mp3",
    "convert_wav_to_pcm",
    "read_wav_metadata",
    "read_wav_pcm",
    "resample_wav",
    "write_wav",
]
