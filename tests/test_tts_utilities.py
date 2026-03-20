import shutil
import tempfile
import unittest
from pathlib import Path

from tts import tts_utilities


def _mono_pcm(sample_value: int, frame_count: int) -> bytes:
    return int(sample_value).to_bytes(2, byteorder="little", signed=True) * frame_count


class AudioDirectoryUtilityTests(unittest.TestCase):
    def test_resample_audio_directory_updates_wav_files_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_path = root / "02_second.wav"
            second_path = root / "01_first.wav"
            tts_utilities.write_wav(_mono_pcm(100, 120), first_path, sample_rate=24000)
            tts_utilities.write_wav(_mono_pcm(200, 80), second_path, sample_rate=24000)

            processed_paths = tts_utilities.resample_audio_directory(root, sample_rate=48000)

            self.assertEqual([path.name for path in processed_paths], ["01_first.wav", "02_second.wav"])
            for path in processed_paths:
                metadata = tts_utilities.read_wav_metadata(path)
                self.assertEqual(metadata.sample_rate, 48000)
                self.assertEqual(metadata.channels, 1)
                self.assertEqual(metadata.sample_width, 2)

    def test_concatenate_audio_directory_uses_sorted_order_and_excludes_existing_total(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_pcm = _mono_pcm(111, 10)
            second_pcm = _mono_pcm(222, 8)
            existing_total_pcm = _mono_pcm(333, 6)
            tts_utilities.write_wav(second_pcm, root / "02_second.wav", sample_rate=24000)
            tts_utilities.write_wav(first_pcm, root / "01_first.wav", sample_rate=24000)
            tts_utilities.write_wav(existing_total_pcm, root / "total.wav", sample_rate=24000)

            output_path = tts_utilities.concatenate_audio_directory(root)
            merged_pcm, spec = tts_utilities.read_wav_pcm(output_path)

            self.assertEqual(output_path.name, "total.wav")
            self.assertEqual(spec.sample_rate, 24000)
            self.assertEqual(spec.channels, 1)
            self.assertEqual(spec.sample_width, 2)
            self.assertEqual(merged_pcm, first_pcm + second_pcm)
            self.assertEqual(len(merged_pcm) // (spec.channels * spec.sample_width), 18)

    def test_empty_directory_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(tts_utilities.AudioUtilityError, "目录中未找到可处理的音频文件"):
                tts_utilities.concatenate_audio_directory(temp_dir)

    def test_pcm_directory_works_with_default_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_pcm = _mono_pcm(10, 12)
            second_pcm = _mono_pcm(20, 6)
            (root / "01_first.pcm").write_bytes(first_pcm)
            (root / "02_second.pcm").write_bytes(second_pcm)

            output_path = tts_utilities.concatenate_audio_directory(root)
            merged_pcm, spec = tts_utilities.read_wav_pcm(output_path)

            self.assertEqual(spec.sample_rate, tts_utilities.DEFAULT_SAMPLE_RATE)
            self.assertEqual(spec.channels, tts_utilities.DEFAULT_CHANNELS)
            self.assertEqual(spec.sample_width, tts_utilities.DEFAULT_SAMPLE_WIDTH)
            self.assertEqual(merged_pcm, first_pcm + second_pcm)

    def test_pcm_directory_works_with_explicit_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_pcm = _mono_pcm(10, 12)
            second_pcm = _mono_pcm(20, 6)
            (root / "01_first.pcm").write_bytes(first_pcm)
            (root / "02_second.pcm").write_bytes(second_pcm)

            output_path = tts_utilities.concatenate_audio_directory(
                root,
                pcm_sample_rate=8000,
                pcm_channels=1,
                pcm_sample_width=2,
            )
            metadata = tts_utilities.read_wav_metadata(output_path)

            self.assertEqual(metadata.sample_rate, 8000)
            self.assertEqual(metadata.channels, 1)
            self.assertEqual(metadata.sample_width, 2)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg 未安装")
    def test_resample_audio_directory_supports_mp3(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_wav = root / "source.wav"
            source_mp3 = root / "source.mp3"
            decoded_wav = root / "decoded.wav"
            tts_utilities.write_wav(_mono_pcm(123, 240), source_wav, sample_rate=24000)
            tts_utilities.convert_wav_to_mp3(source_wav, source_mp3)
            source_wav.unlink()

            processed_paths = tts_utilities.resample_audio_directory(root, sample_rate=48000)
            tts_utilities.convert_mp3_to_wav(source_mp3, decoded_wav, sample_rate=48000)
            metadata = tts_utilities.read_wav_metadata(decoded_wav)

            self.assertEqual([path.name for path in processed_paths], ["source.mp3"])
            self.assertEqual(metadata.sample_rate, 48000)
            self.assertEqual(metadata.channels, 1)
            self.assertEqual(metadata.sample_width, 2)


if __name__ == "__main__":
    unittest.main()
