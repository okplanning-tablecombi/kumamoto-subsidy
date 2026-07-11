# ミラーウィスパー (Mirror Whisper)

ネットワーク制限環境(Claude Code リモート実行環境など)で動く
**完全ローカルの音声文字起こしツール**。

## 背景

この環境の egress プロキシは、Whisper モデルの通常の配布経路
(huggingface.co、openaipublic.azureedge.net、GitHub Releases、
Docker Hub の blob CDN 等)をすべて遮断している。

一方で次は到達可能:

| 経路 | 用途 |
|---|---|
| PyPI (`pypi.org` / `files.pythonhosted.org`) | pywhispercpp / av のインストール |
| `mirror.gcr.io` (Google の Docker Hub ミラー) | モデル同梱イメージのレイヤー取得 |
| `raw.githubusercontent.com` | tiny モデル (動作確認用) |

ミラーウィスパーは、Docker Hub イメージ **`hognir/whisper-cpp:latest`** に
同梱された `ggml-large-v3.bin`(Whisper 最高精度クラス・3.1GB)を
`mirror.gcr.io` 経由でイメージレイヤーごとストリーム取得し、tar から
モデルだけを抽出してローカルの whisper.cpp (pywhispercpp) で推論する。

## 使い方

```bash
# 高精度 (既定: whisper large-v3)
python3 scripts/mirror-whisper/transcribe.py 音声.mp3 --lang ja

# 動作確認用の軽量モデル
python3 scripts/mirror-whisper/transcribe.py 音声.mp3 --model tiny

# 出力先を指定
python3 scripts/mirror-whisper/transcribe.py 音声.mp3 --out 結果.md
```

- 入力は MP3/WAV/M4A/MP4 など ffmpeg (PyAV) が読める形式なら何でも可。
- モデルは `~/.cache/mirror-whisper/` にキャッシュされ、2回目以降は即実行。
- 出力はタイムスタンプ付きの Markdown(表+全文)。

## 目安

| 項目 | 値 |
|---|---|
| モデル取得 (初回のみ) | 約3GB / 数分 |
| 推論速度 (4コアCPU) | 音声1分あたり約5分 (large-v3) |
| 日本語精度 | large-v3: 実用レベル / tiny: 参考程度 |
