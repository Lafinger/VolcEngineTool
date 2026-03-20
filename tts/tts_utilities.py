# -*- coding: utf-8 -*-
"""通用音频处理工具。

`wav` 和 `pcm` 的处理优先使用标准库；`mp3` 相关转换依赖系统中的 `ffmpeg`。
原始 `pcm` 默认按照 16-bit little-endian、单声道、24000 Hz 处理，可按需覆盖。
"""

from __future__ import annotations

import argparse
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
    "SUPPORTED_AUDIO_FORMATS",
    "concatenate_audio_directory",
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
    "resample_audio_directory",
    "resample_wav",
    "write_wav",
]
