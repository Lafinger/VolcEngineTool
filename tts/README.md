# VolcEngineTool TTS

基于火山引擎豆包语音合成 HTTP 单向流式接口的 Python 工具集，提供单条文本合成、按 Markdown 批量导出章节音频，以及 `wav` / `pcm` / `mp3` 的基础音频处理能力。

当前仓库中，推荐的主入口是 [`tts_https.py`](./tts_https.py)；[`demo`](./demo) 目录保留了更贴近原始接口的 HTTP / SSE 示例，适合排障或二次开发时对照使用。

## 功能特性

- 支持 `v1` / `v2` 两套火山引擎 TTS 预设模型
- 通过流式 HTTP 响应拼接 PCM 音频，并统一输出为 `.wav`
- 支持按 Markdown 一级标题批量生成章节音频
- 自动清洗非法文件名字符，遇到重名时自动追加序号
- 提供 `wav` / `pcm` / `mp3` 互转、重采样、拼接等通用工具
- 保留原始 HTTP Chunked 与 SSE 调用示例，便于调试底层接口

## 目录结构

```text
tts/
|-- README.md
|-- tts_https.py
|-- tts_utilities.py
|-- 音频文本.md
|-- demo/
|   |-- tts_http_demo.py
|   `-- tts_http_sse_demo.py
|-- materials/
|   |-- 音频文本案例_中文.md
|   |-- 音频文本案例_英文.md
|   |-- IanFlemingFromRussiaWithLoveUTF8.txt
|   `-- 平凡的世界UTF8.txt
`-- tts_https_wavs/
    `-- v1/
        `-- 主页面.wav
```

各文件职责如下：

- [`tts_https.py`](./tts_https.py)：主入口，负责单条 TTS、Markdown 批量解析与 WAV 输出
- [`tts_utilities.py`](./tts_utilities.py)：音频工具模块，负责元数据读取、格式转换、拼接与重采样
- [`音频文本.md`](./音频文本.md)：默认批量输入文件
- [`materials`](./materials)：额外的中英文样例文本
- [`demo`](./demo)：更接近底层接口请求格式的参考实现
- [`tts_https_wavs`](./tts_https_wavs)：仓库内已有的示例输出目录

## 环境要求

- Python `>= 3.11`
- 依赖包：`requests`
- 如需进行 MP3 相关转换，需安装 `ffmpeg` 并确保已加入 `PATH`
- 有效的火山引擎 TTS 凭证

## 安装

以下命令默认在仓库根目录执行。

使用 `uv`：

```powershell
uv sync
```

如果只想安装最小依赖：

```powershell
python -m pip install requests
```

## 配置凭证

[`tts_https.py`](./tts_https.py) 通过环境变量读取凭证：

```powershell
$env:VOLCENGINE_TTS_APP_ID = "your-app-id"
$env:VOLCENGINE_TTS_ACCESS_KEY = "your-access-key"
$env:VOLCENGINE_TTS_UID = "demo-user"
```

说明：

- `VOLCENGINE_TTS_APP_ID`：必填
- `VOLCENGINE_TTS_ACCESS_KEY`：必填
- `VOLCENGINE_TTS_UID`：可选，默认值为 `codex-tts`

## 模型预设

[`tts_https.py`](./tts_https.py) 内置了两套模型配置：

| 模型 | `resource_id` | 默认 `speaker` |
| --- | --- | --- |
| `v1` | `seed-tts-1.0` | `ICL_zh_female_yry_tob` |
| `v2` | `seed-tts-2.0` | `saturn_zh_female_qingyingduoduo_cs_tob` |

如果你已经有其他可用音色，可以通过命令行参数 `--speaker` 覆盖默认值。

## 快速开始

### 单条文本合成

`v1`：

```powershell
uv run python tts/tts_https.py v1 --text "您好，欢迎光临。" --output tts/tts_https_wavs/v1/demo.wav
```

`v2`：

```powershell
uv run python tts/tts_https.py v2 --text "您好，欢迎光临。" --output tts/tts_https_wavs/v2/demo.wav
```

自定义音色和采样率：

```powershell
uv run python tts/tts_https.py v2 `
  --text "请前往 3 号窗口办理业务。" `
  --output tts/output/guide.wav `
  --speaker saturn_zh_female_qingyingduoduo_cs_tob `
  --sample-rate 24000
```

注意：主脚本始终写出 WAV 文件，建议输出路径使用 `.wav` 扩展名。

### 按 Markdown 批量合成

使用默认输入文件 [`音频文本.md`](./音频文本.md)：

```powershell
uv run python tts/tts_https.py batch --model v1
```

使用自定义输入与输出目录：

```powershell
uv run python tts/tts_https.py batch `
  --model v2 `
  --input tts/materials/音频文本案例_中文.md `
  --output-dir tts/output `
  --sample-rate 24000
```

批量模式会把文件输出到 `<output-dir>/<model>/` 目录，例如 `tts/output/v2/`。

## Markdown 输入格式

批量模式只识别一级标题，即以 `# ` 开头的行。每个标题代表一个输出音频文件，标题下的每一行非空文本会分别请求一次 TTS，最后按顺序拼接成一个 WAV。

示例：

```md
# 主页面
您好，我是您的智能助手。
欢迎来到深圳储蓄银行。

# 取号页面
取号成功！
```

解析规则：

- 只有 `# 标题` 会开启新章节，`##`、`###` 不会被识别
- 空行会被忽略
- 章节标题会被转换为输出文件名
- 文件名中的非法字符会被替换为 `_`
- 如果标题重名，或目标文件已存在，会自动生成 `xxx_2.wav`、`xxx_3.wav`

## 命令参数

### `v1` / `v2`

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--text` | 是 | 待合成文本 |
| `--output` | 是 | 输出 WAV 文件路径 |
| `--speaker` | 否 | 自定义音色 ID |
| `--sample-rate` | 否 | 采样率，默认 `24000` |

### `batch`

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--model` | 是 | 选择批量导出的模型版本，取值为 `v1` 或 `v2` |
| `--input` | 否 | Markdown 输入文件，默认 [`音频文本.md`](./音频文本.md) |
| `--output-dir` | 否 | 输出根目录，默认 [`tts_https_wavs`](./tts_https_wavs) |
| `--sample-rate` | 否 | 采样率，默认 `24000` |

## Python API 用法

除了命令行入口，也可以直接在 Python 中调用。

```python
from tts.tts_https import synthesize_markdown_cases, synthesize_tts_v1
from tts.tts_utilities import concatenate_wav_files, convert_wav_to_mp3

single_path = synthesize_tts_v1(
    "您好，欢迎光临。",
    "tts/tts_https_wavs/v1/demo.wav",
)

batch_paths = synthesize_markdown_cases(
    "tts/materials/音频文本案例_中文.md",
    "tts/output",
    model="v2",
)

convert_wav_to_mp3(single_path, "tts/output/demo.mp3")
concatenate_wav_files(batch_paths[:2], "tts/output/merged.wav")
```

如果要调用 MP3 相关接口，请先确认系统中可执行 `ffmpeg`。

## 音频工具概览

[`tts_utilities.py`](./tts_utilities.py) 目前提供以下能力：

- 读取音频信息：`read_wav_metadata`、`read_wav_pcm`
- 写入 WAV：`write_wav`
- 音频格式转换：`convert_wav_to_pcm`、`convert_pcm_to_wav`、`convert_wav_to_mp3`、`convert_mp3_to_wav`、`convert_pcm_to_mp3`、`convert_mp3_to_pcm`
- 通用转换入口：`convert_audio_format`
- 音频处理：`concatenate_wav_files`、`resample_wav`

实现细节：

- `wav` / `pcm` 的处理主要依赖 Python 标准库
- `mp3` 相关转换通过 `ffmpeg` 完成
- 默认 PCM 规格为 16-bit little-endian、单声道、`24000 Hz`

## Demo 脚本

[`demo/tts_http_demo.py`](./demo/tts_http_demo.py) 和 [`demo/tts_http_sse_demo.py`](./demo/tts_http_sse_demo.py) 是更接近原始接口的示例：

- `tts_http_demo.py`：请求 `https://openspeech.bytedance.com/api/v3/tts/unidirectional`
- `tts_http_sse_demo.py`：请求 `https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse`
- 两个脚本都需要手动填写 `appID`、`accessKey`、`resourceID`
- 两个脚本的请求体都使用 `mp3` 作为返回格式

如果你的目标是快速接入并稳定生成 WAV，优先使用 [`tts_https.py`](./tts_https.py)。

## 常见问题

### 1. 运行时报“缺少环境变量”

请确认已经设置：

- `VOLCENGINE_TTS_APP_ID`
- `VOLCENGINE_TTS_ACCESS_KEY`

`VOLCENGINE_TTS_UID` 可省略。

### 2. 批量模式提示未解析到可合成内容

请检查输入 Markdown 是否满足以下条件：

- 使用 `# ` 开头的一级标题
- 标题下至少有一行非空文本
- 文件保存为 UTF-8 编码

### 3. MP3 转换失败

如果错误信息提示找不到 `ffmpeg`，说明当前系统未安装 `ffmpeg`，或未把它加入 `PATH`。这只会影响 MP3 相关能力，不影响主脚本生成 WAV。

### 4. 输出路径用了 `.mp3` 扩展名但文件无法正常识别

[`tts_https.py`](./tts_https.py) 始终输出 WAV 封装文件，不会根据扩展名自动切换编码格式。请为主脚本输出路径使用 `.wav`。

## License

本项目采用 MIT License，详见 [`../LICENSE`](../LICENSE)。
