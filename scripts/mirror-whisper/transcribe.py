#!/usr/bin/env python3
"""ミラーウィスパー (Mirror Whisper) — ネットワーク制限環境向けローカル文字起こしツール

Hugging Face 等の通常のモデル配布サイトへの通信が遮断された環境でも、
到達可能なミラー経由で Whisper モデルを取得してローカルで文字起こしする。

モデルの取得経路:
  - large-v3 (高精度・既定):
      Docker Hub イメージ hognir/whisper-cpp:latest に同梱された
      ggml-large-v3.bin を、Google の Docker Hub ミラー (mirror.gcr.io)
      経由でイメージレイヤーごとストリーム取得し、tar から直接抽出する。
      (Docker Hub 本体の blob CDN は遮断されていても mirror.gcr.io の
       blob は storage.googleapis.com 配信のため到達可能)
  - tiny (低精度・動作確認用):
      GitHub リポジトリ Macoron/whisper.unity に直接コミットされた
      ggml-tiny.bin を raw.githubusercontent.com から取得する。

使い方:
  python3 transcribe.py 音声ファイル [--lang ja] [--model large-v3|tiny]
                                     [--out 出力.md] [--threads N]

対応入力: MP3 / WAV / M4A / MP4 など PyAV (ffmpeg) が読める形式すべて。
出力: タイムスタンプ付きテキスト (標準出力) と Markdown ファイル。
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

CACHE_DIR = Path(os.environ.get("MIRROR_WHISPER_CACHE", "~/.cache/mirror-whisper")).expanduser()

# Docker Hub 上のモデル同梱イメージ (whisper.cpp + ggml-large-v3.bin 入り)
GCR_MIRROR = "https://mirror.gcr.io"
IMAGE_REPO = "hognir/whisper-cpp"
IMAGE_TAG = "latest"
LAYER_MEMBER = "app/models/ggml-large-v3.bin"
# 2026-07 時点の既知レイヤー。タグ更新で変わったら resolve_layer_digest() が再解決する
KNOWN_LAYER_DIGEST = "sha256:54d1b1f955a743f51106890376db0e7c770590ae8d6506d5ba2cd3c62a7695f7"
MIN_MODEL_LAYER_BYTES = 2_500_000_000  # モデル入りレイヤーは 3GB 級

TINY_URL = "https://raw.githubusercontent.com/Macoron/whisper.unity/master/Assets/StreamingAssets/Whisper/ggml-tiny.bin"

MANIFEST_ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
])


def curl_json(url, headers=None):
    cmd = ["curl", "-sSL", "--max-time", "60", url]
    for k, v in (headers or {}).items():
        cmd += ["-H", f"{k}: {v}"]
    out = subprocess.run(cmd, capture_output=True, timeout=90)
    return json.loads(out.stdout)


def ensure_deps():
    """pywhispercpp / av が無ければ PyPI (直接到達可) から入れる。"""
    missing = []
    for mod, pkg in (("pywhispercpp", "pywhispercpp"), ("av", "av")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[mirror-whisper] 依存パッケージをインストール中: {' '.join(missing)}", file=sys.stderr)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)


def registry_token():
    d = curl_json(f"{GCR_MIRROR}/v2/token?service=mirror.gcr.io&scope=repository:{IMAGE_REPO}:pull")
    return d.get("token") or d.get("access_token")


def resolve_layer_digest(token):
    """イメージマニフェストからモデル入り (>2.5GB) レイヤーの digest を得る。"""
    man = curl_json(
        f"{GCR_MIRROR}/v2/{IMAGE_REPO}/manifests/{IMAGE_TAG}",
        {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
    )
    if "manifests" in man:  # マルチアーキの index → amd64 を選ぶ
        digest = next(m["digest"] for m in man["manifests"]
                      if m.get("platform", {}).get("architecture") == "amd64")
        man = curl_json(
            f"{GCR_MIRROR}/v2/{IMAGE_REPO}/manifests/{digest}",
            {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
        )
    for layer in man["layers"]:
        if layer["size"] > MIN_MODEL_LAYER_BYTES:
            return layer["digest"]
    raise RuntimeError("モデル入りレイヤーが見つかりません (イメージ構成が変わった可能性)")


def blob_reachable(token, digest):
    code = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "30",
         "-r", "0-1023", "-L", "-H", f"Authorization: Bearer {token}",
         f"{GCR_MIRROR}/v2/{IMAGE_REPO}/blobs/{digest}"],
        capture_output=True, text=True, timeout=60,
    ).stdout.strip()
    return code in ("200", "206")


def download_large_v3(dest: Path):
    """レイヤー tar.gz をストリームしながら ggml-large-v3.bin だけを抽出する。
    (3GB の tar.gz をディスクに置かずに済む)"""
    token = registry_token()
    digest = KNOWN_LAYER_DIGEST
    if not blob_reachable(token, digest):
        print("[mirror-whisper] 既知レイヤーが見つからないため再解決します", file=sys.stderr)
        digest = resolve_layer_digest(token)
    print(f"[mirror-whisper] モデルレイヤーをストリーム取得中 (約3GB): {digest[:20]}…", file=sys.stderr)

    proc = subprocess.Popen(
        ["curl", "-sSL", "--max-time", "1800", "-H", f"Authorization: Bearer {token}",
         f"{GCR_MIRROR}/v2/{IMAGE_REPO}/blobs/{digest}"],
        stdout=subprocess.PIPE,
    )
    tmp = dest.with_suffix(".part")
    found = False
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tf:
            for member in tf:
                if member.name.lstrip("./") == LAYER_MEMBER:
                    src = tf.extractfile(member)
                    with open(tmp, "wb") as out:
                        while True:
                            chunk = src.read(1 << 22)
                            if not chunk:
                                break
                            out.write(chunk)
                    found = True
                    break
    finally:
        proc.stdout.close()
        proc.terminate()
        proc.wait()
    if not found:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"レイヤー内に {LAYER_MEMBER} が見つかりませんでした")
    tmp.rename(dest)


def download_tiny(dest: Path):
    print("[mirror-whisper] tiny モデルを取得中 (78MB)…", file=sys.stderr)
    tmp = dest.with_suffix(".part")
    subprocess.run(["curl", "-sSL", "--max-time", "600", "-o", str(tmp), TINY_URL], check=True)
    tmp.rename(dest)


def ensure_model(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / f"ggml-{name}.bin"
    expected_min = {"large-v3": 3_000_000_000, "tiny": 70_000_000}[name]
    if dest.exists() and dest.stat().st_size >= expected_min:
        return dest
    dest.unlink(missing_ok=True)
    if name == "large-v3":
        download_large_v3(dest)
    else:
        download_tiny(dest)
    size = dest.stat().st_size
    if size < expected_min:
        raise RuntimeError(f"モデルの取得に失敗 (サイズ異常 {size} bytes)")
    print(f"[mirror-whisper] モデル取得完了: {dest} ({size/1e9:.2f} GB)", file=sys.stderr)
    return dest


def to_wav16k(audio_path: Path, wav_path: Path):
    """16kHz モノラルに変換し、ピーク正規化 (0.95) して書き出す。"""
    import av
    import numpy as np
    import wave
    container = av.open(str(audio_path))
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    for frame in container.decode(stream):
        for f in resampler.resample(frame):
            chunks.append(f.to_ndarray().flatten())
    x = np.concatenate(chunks).astype(np.float32)
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (0.95 * 32767 / peak)
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.clip(x, -32768, 32767).astype(np.int16).tobytes())


def fmt_ts(centisec: float) -> str:
    total = int(centisec / 100)
    return f"{total // 60}:{total % 60:02d}"


def main():
    p = argparse.ArgumentParser(description="ミラーウィスパー: 制限環境向けローカル文字起こし")
    p.add_argument("audio", help="入力音声/動画ファイル")
    p.add_argument("--lang", default="ja", help="言語コード (既定: ja)")
    p.add_argument("--model", default="large-v3", choices=["large-v3", "tiny"],
                   help="large-v3=高精度(既定・3GB) / tiny=動作確認用(78MB)")
    p.add_argument("--out", default=None, help="出力 Markdown パス (既定: <入力名>.transcript.md)")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--keep-context", action="store_true",
                   help="30秒窓間で文脈を引き継ぐ (既定は無効。有効にすると静かな音声で"
                        "同一文の反復ハルシネーションが起きることがある。注意: whisper.cpp の"
                        "no_context は窓間の文脈引き継ぎを止めないため、n_max_text_ctx=0 で切る)")
    args = p.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        sys.exit(f"入力ファイルがありません: {audio}")

    ensure_deps()
    model_path = ensure_model(args.model)

    wav = Path(f"/tmp/mirror-whisper-{os.getpid()}.wav")
    try:
        print("[mirror-whisper] 音声を 16kHz WAV に変換中…", file=sys.stderr)
        to_wav16k(audio, wav)

        print(f"[mirror-whisper] 文字起こし中 (model={args.model}, threads={args.threads})…", file=sys.stderr)
        from pywhispercpp.model import Model
        # n_max_text_ctx=0 が窓間の文脈引き継ぎを完全に断つ唯一のノブ。
        # (no_context=True は whisper_full 冒頭で履歴を消すだけで、同一呼び出し内の
        #  窓間では prompt_past1 が常に引き継がれ、静かな窓の幻覚が全体に伝搬する)
        ctx_params = {} if args.keep_context else {"n_max_text_ctx": 0}
        m = Model(str(model_path), n_threads=args.threads, **ctx_params)
        segments = m.transcribe(str(wav), language=args.lang)
    finally:
        wav.unlink(missing_ok=True)

    out_path = Path(args.out) if args.out else Path.cwd() / f"{audio.stem}.transcript.md"
    lines = [f"# 文字起こし: {audio.name}", "",
             f"- モデル: whisper {args.model} (ミラーウィスパー)", "", "| 時間 | テキスト |", "|---|---|"]
    for s in segments:
        text = s.text.strip()
        if text:
            print(f"[{fmt_ts(s.t0)}–{fmt_ts(s.t1)}] {text}")
            lines.append(f"| {fmt_ts(s.t0)}–{fmt_ts(s.t1)} | {text} |")
    lines += ["", "## 全文", "", "".join(s.text.strip() for s in segments), ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[mirror-whisper] 保存しました: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
