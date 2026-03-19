import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tts import tts_https


class BatchSynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.credentials = tts_https._Credentials(app_id="app", access_key="key", uid="uid")

    def _write_markdown(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\r\n")

    def test_skip_existing_file_in_current_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "input.md"
            output_root = root / "output"
            existing_path = output_root / "v2" / "示例标题.wav"
            existing_path.parent.mkdir(parents=True, exist_ok=True)
            existing_path.write_bytes(b"existing")
            self._write_markdown(markdown_path, "# 示例标题\r\n已有音频时应跳过\r\n")

            with patch.object(tts_https, "_load_credentials", return_value=self.credentials), patch.object(
                tts_https,
                "_combine_section_pcm",
                side_effect=AssertionError("existing files should be skipped before synthesis"),
            ):
                generated_paths = tts_https.synthesize_markdown_cases(markdown_path, output_root, model="v2")

            self.assertEqual(generated_paths, [])
            self.assertEqual(existing_path.read_bytes(), b"existing")

    def test_generate_fixed_filename_without_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "input.md"
            output_root = root / "output"
            self._write_markdown(markdown_path, "# 标题/带非法字符\r\n生成新的音频文件\r\n")
            expected_path = output_root / "v2" / "标题_带非法字符.wav"

            with patch.object(tts_https, "_load_credentials", return_value=self.credentials), patch.object(
                tts_https,
                "_combine_section_pcm",
                return_value=b"\x00\x00\x01\x00",
            ) as combine_mock:
                generated_paths = tts_https.synthesize_markdown_cases(markdown_path, output_root, model="v2")

            self.assertEqual(generated_paths, [expected_path])
            self.assertTrue(expected_path.exists())
            self.assertFalse((output_root / "v2" / "标题_带非法字符_2.wav").exists())
            self.assertEqual(combine_mock.call_count, 1)

    def test_duplicate_titles_in_same_markdown_are_skipped_after_first_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "input.md"
            output_root = root / "output"
            self._write_markdown(markdown_path, "# 重复标题\r\n第一段\r\n\r\n# 重复标题\r\n第二段\r\n")
            expected_path = output_root / "v2" / "重复标题.wav"

            with patch.object(tts_https, "_load_credentials", return_value=self.credentials), patch.object(
                tts_https,
                "_combine_section_pcm",
                return_value=b"\x00\x00\x01\x00",
            ) as combine_mock:
                generated_paths = tts_https.synthesize_markdown_cases(markdown_path, output_root, model="v2")

            self.assertEqual(generated_paths, [expected_path])
            self.assertTrue(expected_path.exists())
            self.assertFalse((output_root / "v2" / "重复标题_2.wav").exists())
            self.assertEqual(combine_mock.call_count, 1)

    def test_other_model_directory_does_not_block_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_path = root / "input.md"
            output_root = root / "output"
            v1_existing_path = output_root / "v1" / "跨模型同名.wav"
            v1_existing_path.parent.mkdir(parents=True, exist_ok=True)
            v1_existing_path.write_bytes(b"existing")
            self._write_markdown(markdown_path, "# 跨模型同名\r\n当前模型仍应生成\r\n")
            expected_v2_path = output_root / "v2" / "跨模型同名.wav"

            with patch.object(tts_https, "_load_credentials", return_value=self.credentials), patch.object(
                tts_https,
                "_combine_section_pcm",
                return_value=b"\x00\x00\x01\x00",
            ) as combine_mock:
                generated_paths = tts_https.synthesize_markdown_cases(markdown_path, output_root, model="v2")

            self.assertEqual(generated_paths, [expected_v2_path])
            self.assertTrue(expected_v2_path.exists())
            self.assertEqual(v1_existing_path.read_bytes(), b"existing")
            self.assertEqual(combine_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
