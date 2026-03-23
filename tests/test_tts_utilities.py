import contextlib
import io
import math
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from tts import tts_utilities

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _build_tone_pcm(
    duration_seconds: float,
    *,
    sample_rate: int = tts_utilities.DEFAULT_SAMPLE_RATE,
    amplitude: int = 12000,
    frequency: float = 440.0,
) -> bytes:
    frame_count = int(round(duration_seconds * sample_rate))
    pcm = bytearray()
    for index in range(frame_count):
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * (index / sample_rate)))
        pcm.extend(sample.to_bytes(2, byteorder="little", signed=True))
    return bytes(pcm)


def _build_silence_pcm(
    duration_seconds: float,
    *,
    sample_rate: int = tts_utilities.DEFAULT_SAMPLE_RATE,
) -> bytes:
    frame_count = int(round(duration_seconds * sample_rate))
    return b"\x00\x00" * frame_count


class TtsUtilitiesTests(unittest.TestCase):
    def _write_wav(self, path: Path, pcm_data: bytes, *, sample_rate: int = tts_utilities.DEFAULT_SAMPLE_RATE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(tts_utilities.DEFAULT_CHANNELS)
            wav_file.setsampwidth(tts_utilities.DEFAULT_SAMPLE_WIDTH)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

    def test_detect_audio_language_from_path_and_override(self) -> None:
        self.assertEqual(tts_utilities.detect_audio_language("tts/tts_https_wavs/total/total_en.wav"), "en")
        self.assertEqual(tts_utilities.detect_audio_language("tts/tts_https_wavs/total/total_cn.wav"), "zh")
        self.assertEqual(tts_utilities.detect_audio_language("tts/tts_https_wavs/中文.wav"), "zh")
        self.assertEqual(tts_utilities.detect_audio_language("tts/tts_https_wavs/unknown.wav", language="英文"), "en")

        with self.assertRaisesRegex(ValueError, "无法根据音频路径自动识别语言"):
            tts_utilities.detect_audio_language("tts/tts_https_wavs/unknown.wav")

        with self.assertRaisesRegex(ValueError, "同时包含中文和英文"):
            tts_utilities.detect_audio_language("tts/tts_https_wavs/total/total_cn_en.wav")

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg is required for silence detection tests")
    def test_split_wav_on_pauses_prefers_silence_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "total_en.wav"
            pcm_data = b"".join(
                (
                    _build_tone_pcm(4.5),
                    _build_silence_pcm(0.4),
                    _build_tone_pcm(4.6),
                    _build_silence_pcm(0.35),
                    _build_tone_pcm(4.4),
                )
            )
            self._write_wav(input_path, pcm_data)

            result = tts_utilities.split_wav_on_pauses(input_path, None, "DuoDuo")

            self.assertEqual(result.language, "en")
            self.assertEqual(result.output_dir, root / "DuoDuo")
            self.assertEqual([path.name for path in result.output_paths], ["DuoDuo_0.wav", "DuoDuo_1.wav", "DuoDuo_2.wav"])
            self.assertEqual(len(result.durations), 3)
            self.assertTrue(all(2.0 <= duration <= 7.0 for duration in result.durations))
            self.assertAlmostEqual(result.durations[0], 4.7, delta=0.35)
            self.assertAlmostEqual(result.durations[1], 4.775, delta=0.35)
            self.assertAlmostEqual(result.durations[2], 4.375, delta=0.35)
            self.assertTrue(all(path.is_file() for path in result.output_paths))

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg is required for silence detection tests")
    def test_split_wav_on_pauses_uses_low_energy_fallback_without_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "total_en.wav"
            self._write_wav(input_path, _build_tone_pcm(15.0))

            result = tts_utilities.split_wav_on_pauses(input_path, None, "DuoDuo")

            self.assertEqual([path.name for path in result.output_paths], ["DuoDuo_0.wav", "DuoDuo_1.wav", "DuoDuo_2.wav"])
            self.assertEqual(len(result.durations), 3)
            self.assertTrue(all(2.0 <= duration <= 7.0 for duration in result.durations))
            for duration in result.durations:
                self.assertAlmostEqual(duration, 5.0, delta=0.1)

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg is required for silence detection tests")
    def test_split_wav_on_pauses_appends_after_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "DuoDuo"
            output_dir.mkdir(parents=True, exist_ok=True)
            self._write_wav(output_dir / "DuoDuo_0.wav", _build_tone_pcm(2.5))
            self._write_wav(output_dir / "DuoDuo_1.wav", _build_tone_pcm(2.5))

            input_path = root / "total_en.wav"
            pcm_data = b"".join(
                (
                    _build_tone_pcm(4.5),
                    _build_silence_pcm(0.35),
                    _build_tone_pcm(4.5),
                )
            )
            self._write_wav(input_path, pcm_data)

            result = tts_utilities.split_wav_on_pauses(input_path, output_dir, "DuoDuo")

            self.assertEqual([path.name for path in result.output_paths], ["DuoDuo_2.wav", "DuoDuo_3.wav"])

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg is required for silence detection tests")
    def test_split_wavs_on_pauses_assigns_continuous_indices_across_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_input = root / "total_en.wav"
            second_input = root / "total_cn.wav"
            shared_output_dir = root / "DuoDuo"
            first_pcm = b"".join(
                (
                    _build_tone_pcm(4.5),
                    _build_silence_pcm(0.35),
                    _build_tone_pcm(4.5),
                )
            )
            second_pcm = b"".join(
                (
                    _build_tone_pcm(4.4),
                    _build_silence_pcm(0.4),
                    _build_tone_pcm(4.6),
                )
            )
            self._write_wav(first_input, first_pcm)
            self._write_wav(second_input, second_pcm)

            result = tts_utilities.split_wavs_on_pauses([first_input, second_input], shared_output_dir, "DuoDuo")

            self.assertEqual(result.output_dir, shared_output_dir)
            self.assertEqual(len(result.results), 2)
            self.assertEqual(result.results[0].language, "en")
            self.assertEqual(result.results[1].language, "zh")
            self.assertEqual(
                [path.name for path in result.results[0].output_paths],
                ["DuoDuo_0.wav", "DuoDuo_1.wav"],
            )
            self.assertEqual(
                [path.name for path in result.results[1].output_paths],
                ["DuoDuo_2.wav", "DuoDuo_3.wav"],
            )

    @unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg is required for silence detection tests")
    def test_split_wav_cli_reports_language_and_filenames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_input = root / "total_en.wav"
            second_input = root / "total_cn.wav"
            output_dir = root / "clips"
            first_pcm = b"".join(
                (
                    _build_tone_pcm(4.5),
                    _build_silence_pcm(0.35),
                    _build_tone_pcm(4.5),
                )
            )
            second_pcm = b"".join(
                (
                    _build_tone_pcm(4.4),
                    _build_silence_pcm(0.4),
                    _build_tone_pcm(4.6),
                )
            )
            self._write_wav(first_input, first_pcm)
            self._write_wav(second_input, second_pcm)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = tts_utilities.main(
                    [
                        "split-wav",
                        "--input-file",
                        str(first_input),
                        "--input-file",
                        str(second_input),
                        "--output-dir",
                        str(output_dir),
                        "--split-name",
                        "DuoDuo",
                    ]
                )

            self.assertEqual(exit_code, 0)
            output_text = stdout.getvalue()
            self.assertIn("输入文件数：2", output_text)
            self.assertIn("总生成文件数：4", output_text)
            self.assertIn(f"输入文件：{first_input}", output_text)
            self.assertIn(f"输入文件：{second_input}", output_text)
            self.assertIn("语言类型：en", output_text)
            self.assertIn("语言类型：zh", output_text)
            self.assertIn("DuoDuo_0.wav", output_text)
            self.assertIn("DuoDuo_1.wav", output_text)
            self.assertIn("DuoDuo_2.wav", output_text)
            self.assertIn("DuoDuo_3.wav", output_text)


if __name__ == "__main__":
    unittest.main()
