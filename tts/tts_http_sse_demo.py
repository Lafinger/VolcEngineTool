# -*- coding: utf-8 -*-
# @Project : tob_service
# @Company : ByteDance
# @Time    : 2026/2/27 10:00
# @Author  : SiNian
# @FileName: TTSv3HttpDemo.py
# @IDE: PyCharm
# @Motto：  I,with no mountain to rely on,am the mountain myself.
import requests
import json
import base64
import os
import traceback

# python版本：==3.11

# -------------客户需要填写的参数----------------
appID = ""
accessKey = ""
resourceID = ""

text = "这是一段测试文本，用于测试字节大模型语音合成http单向流式接口效果。"
# ---------------请求地址----------------------
url = "https://openspeech.bytedance.com/api/v3/tts/unidirectional/sse"

def parse_event(stream):
    event = {
        "event": "",
        "data": ""
    }

    for raw_line in stream:
        line = raw_line.decode("utf-8").strip()
        # 空行 = 一个完整事件结束
        if line == "":
            if event["data"]:
                # 去掉最后一个换行
                event["data"] = event["data"].rstrip("\n")
                yield event
            event = {
                "id": None,
                "event": "message",
                "data": "",
                "retry": None
            }
            continue

        # 注释行（以:开头）
        if line.startswith(":"):
            continue

        if ":" in line:
            field, value = line.split(":", 1)
            value = value.lstrip()

            if field == "data":
                event["data"] += value + "\n"
            elif field == "event":
                event["event"] = value

    # 处理流结束但没有空行的情况
    if event["data"]:
        event["data"] = event["data"].rstrip("\n")
        yield event

def tts_http_sse_stream(url, headers, params, audio_save_path):
    session = requests.Session()
    try:
        print('请求的url:', url)
        print('请求的headers:', headers)
        print('请求的params:\n', params)
        response = session.post(url, headers=headers, json=params, stream=True)
        print(response)
        # 打印response headers
        print(f"code: {response.status_code} header: {response.headers}")
        logid = response.headers.get('X-Tt-Logid')
        print(f"X-Tt-Logid: {logid}")

        # 用于存储音频数据
        audio_data = bytearray()
        total_audio_size = 0
        for event_data in parse_event(response.iter_lines()):
            if not event_data:
                continue
            print('get event', event_data['event'])
            data = json.loads(event_data['data'])

            if data.get("code", 0) == 0 and "data" in data and data["data"]:
                chunk_audio = base64.b64decode(data["data"])
                audio_size = len(chunk_audio)
                total_audio_size += audio_size
                audio_data.extend(chunk_audio)
                continue
            if data.get("code", 0) == 0 and "sentence" in data and data["sentence"]:
                print("sentence_data:", data)
                continue
            if data.get("code", 0) == 20000000:
                if 'usage' in data:
                    print("usage:", data['usage'])
                break
            if data.get("code", 0) > 0:
                print(f"error response:{data}")
                break

        # 保存音频文件
        if audio_data:
            with open(audio_save_path, "wb") as f:
                f.write(audio_data)
            print(f"文件保存在{audio_save_path},文件大小: {len(audio_data) / 1024:.2f} KB")
            # 确保生成的音频有正确的访问权限
            os.chmod(audio_save_path, 0o644)

    except Exception as e:
        print(f"请求失败: {e}")
        traceback.print_exc()
    finally:
        response.close()
        session.close()

if __name__ == "__main__":
    # ---------------请求地址----------------------
    headers = {
        "X-Api-App-Id": appID,
        "X-Api-Access-Key": accessKey,
        "X-Api-Resource-Id": resourceID,
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        
        # 表示是否需要用量返回, 默认不添加; 启用后在合成结束时会多一个usage字段
        # "X-Control-Require-Usage-Tokens-Return": "*" 
    }

    payload = {
        "user": {
            "uid": "123123"
        },
        "req_params":{
            "text": "其他人",
            "speaker": "zh_female_cancan_mars_bigtts",
            "audio_params": {
                "format": "mp3",
                "sample_rate": 24000,
                "enable_timestamp": True
            },
            "additions": "{\"explicit_language\":\"zh\",\"disable_markdown_filter\":true, \"enable_timestamp\":true}\"}"
        }
    }

    tts_http_sse_stream(url=url, headers=headers, params=payload, audio_save_path="tts_test.mp3")
