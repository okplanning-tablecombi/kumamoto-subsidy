---
name: mirror-whisper
description: 音声・動画ファイルの文字起こし(ミラーウィスパー)。ユーザーが音声ファイル(MP3/WAV/M4A/MP4等)をアップロードして「文字起こしして」と依頼したときに必ず使う。Hugging Face等が遮断された環境でも到達可能なミラー経由でWhisperモデルを取得し、ローカルで高精度に文字起こしする。
---

# ミラーウィスパー (Mirror Whisper)

ネットワーク制限のあるリモート実行環境で音声ファイルを文字起こしするための仕組み。
Whisper large-v3 を、遮断されていない経路(Google の Docker Hub ミラー
`mirror.gcr.io` → `storage.googleapis.com`)から取得してローカル実行する。

## 実行手順

1. アップロードされた音声ファイルのパスを確認する(通常 `/root/.claude/uploads/...`)。

2. 次を実行する:

   ```bash
   python3 scripts/mirror-whisper/transcribe.py <音声ファイル> --lang ja
   ```

   - 初回はモデル取得(約3GB、数分)+依存インストールが走る。2回目以降は
     `~/.cache/mirror-whisper/` のキャッシュを使うので即開始される。
   - 61秒の音声で large-v3 の推論に4コアCPUで5分程度かかる。長い音声は
     `run_in_background: true` で実行し、完了通知を待つこと。
   - 動作確認だけなら `--model tiny`(78MB・低精度)が使える。

3. 出力(タイムスタンプ付き Markdown: `<入力名>.transcript.md`)を整形して
   ユーザーに届ける。SendUserFile でファイルも添付する。

## 結果を整形するときの注意

- 固有名詞(人名など)は誤認識されやすい。文脈から明らかな場合は補正し、
  補正したことを必ず注記する。
- 末尾や無音・BGM区間で Whisper は文を捏造(ハルシネーション)することが
  ある。前後の文脈から浮いた唐突な文は除外し、注記する。
- 話者が複数いる場合は改行や「——」で応答を分けると読みやすい。

## 環境の前提と代替経路(トラブルシューティング)

この環境の egress プロキシは huggingface.co / openaipublic.azureedge.net /
GitHub Releases などのモデル配布経路を遮断している。到達可能なのは
パッケージレジストリ(PyPI/npm/Maven/conda 等)、GitHub の
raw/media.githubusercontent.com、gitlab.com、mirror.gcr.io/ghcr.io などの
コンテナレジストリ、storage.googleapis.com。

- Docker Hub 本体の blob CDN (cloudfront.docker.com) は遮断されているが、
  `mirror.gcr.io`(Docker Hub のプルスルーキャッシュ)の blob は
  storage.googleapis.com 配信のため取得できる — これが本スキルの中核。
- イメージ `hognir/whisper-cpp:latest` の 3GB レイヤーに
  `app/models/ggml-large-v3.bin` が同梱されている。タグが更新された場合、
  スクリプトはマニフェストを再解決して 2.5GB 超のレイヤーを自動選択する。
- それでも壊れた場合の代替: Docker Hub 検索 API (hub.docker.com、到達可) で
  モデル同梱イメージを探し直す。イメージの見分け方: マニフェストのレイヤーに
  モデルサイズ級 (数百MB〜数GB) のものがあり、config の履歴に
  download-ggml-model 等の痕跡がある。
- 日本語特化の代替モデル: `alphacep/kaldi-ja:latest`(Docker Hub)の 1GB
  レイヤーに vosk-model-ja-0.22(CER 8-9%)が入っており、pip の vosk で
  動かせる(vosk は依存 srt のビルドが壊れるため `pip install --no-deps vosk`
  + cffi/requests/tqdm/websockets を個別に入れ、`sys.modules['srt']` を
  ダミー登録してから import する)。
