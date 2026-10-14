#!/usr/bin/env python3
"""Transcribe 16 kHz mono WAV files with asr4all-m, its streaming acoustic model running on the Pi's GPU.

usage: transcribe.py clip.wav [more.wav ...]      (--cpu: plain onnxruntime instead, to compare)
Punctuation (PCEC) and decoding run on the CPU through FUTO's own driver, onnx/asr4all_onnx.py.
"""
import importlib.util
import sys
import time
import wave

import numpy as np

import a4a_gpu


def load_driver():
    spec = importlib.util.spec_from_file_location("asr4all_onnx", a4a_gpu.ONNX_DIR + "/asr4all_onnx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16 kHz mono 16-bit WAV (ffmpeg -i in -ar 16000 -ac 1 out.wav)")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def main():
    args = [a for a in sys.argv[1:] if a != "--cpu"]
    if not args:
        sys.exit(__doc__)
    engine = load_driver().load_engine(a4a_gpu.ONNX_DIR)
    engine.set_tier("stream_c16r4")
    if "--cpu" not in sys.argv:
        t = time.perf_counter()
        a4a_gpu.use_gpu(engine)
        print(f"[model on the GPU in {time.perf_counter() - t:.1f} s]", file=sys.stderr)
    for path in args:
        pcm = read_wav(path)
        t, c = time.perf_counter(), time.process_time()
        text = engine.decode_offline(pcm)
        wall, cpu = time.perf_counter() - t, time.process_time() - c
        print(f"[{path}: {len(pcm) / 16000:.1f} s audio, {wall:.1f} s, {cpu:.1f} s CPU]", file=sys.stderr)
        print(text, flush=True)


if __name__ == "__main__":
    main()
