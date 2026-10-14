# asr4all-m speech-to-text on the Raspberry Pi 4 / 400 GPU

Runs the streaming acoustic model of FUTO's [asr4all-m](https://huggingface.co/futo-org/asr4all-m)
speech-to-text model on the Pi's own GPU (VideoCore VI, V3D 4.2) through Vulkan, with hand-written compute
shaders. The words come out the same as the normal CPU version, but the CPU is left almost idle.

The GPU is **not faster** than the CPU here. What you get is a free CPU: dictation keeps up with speech
while the CPU stays available for everything else.

## Results (Pi 400, GPU at the default 500 MHz)

| | CPU (onnxruntime) | GPU (this repo) |
|---|---|---|
| Text on 32 real dictation clips (218 s of speech) | — | identical to the CPU on 32/32 |
| Time for those 218 s of speech | ~130 s | ~190 s (0.86–0.89x real time) |
| CPU time used for them | ~500 s | ~46 s |
| CPU while you talk (live dictation, 2 threads) | ~1 core | ~0.15 core |
| One model step (0.64 s of audio) | 359 ms (3 threads) | ~510 ms |

Accuracy check: log-probabilities within ~3e-5 of onnxruntime, the same best token on every frame.
Raising the GPU clock to 600 MHz (`v3d_freq=600` in config.txt) gave about 5–7% more speed; 550 MHz gave nothing.

## What you need

- A Raspberry Pi 4 or Pi 400. The shaders are tuned for its GPU (V3D 4.2). Pi 5 has a different GPU: untested.
- A 64-bit OS with Mesa's `v3dv` Vulkan driver. **Raspberry Pi OS (64-bit, Bookworm) ships it.**
- About 1.1 GB of free RAM: that is the peak while the model is converted at startup (the CPU-only
  version peaks at about 0.4 GB). A 4 GB Pi is fine; a 2 GB one is tight.

Tested on a Pi 400 running Gentoo with Raspberry Pi OS's Vulkan driver (Mesa 25.0.7). Not yet tested on a
fresh Raspberry Pi OS install; the steps below are what it should take.

## Setup

```sh
sudo apt install libvulkan-dev mesa-vulkan-drivers python3-venv
git clone https://github.com/tarelerulz/pi-gpu-asr4all
cd pi-gpu-asr4all
make lib                       # builds libygpu.so; the shaders come prebuilt in shaders/*.spv

python3 -m venv venv
venv/bin/pip install -r requirements.txt huggingface_hub

# FUTO's model (about 210 MB), into ./asr4all-m/onnx — read its LICENSE, see NOTICE.md
venv/bin/hf download futo-org/asr4all-m --include "onnx/*" --local-dir asr4all-m
venv/bin/hf download futo-org/asr4all-m LICENSE --local-dir asr4all-m
```

(Older `huggingface_hub` versions call the command `huggingface-cli download` instead of `hf download`.)

Model somewhere else? `export A4A_MODEL_DIR=/path/to/asr4all-m/onnx`.

To change a shader you also need `glslc` (`sudo apt install glslc`); then `make` rebuilds everything.

## Use

Transcribe WAV files (16 kHz, mono, 16-bit):

```sh
venv/bin/python transcribe.py clip.wav
venv/bin/python transcribe.py --cpu clip.wav      # the normal CPU version, to compare
```

Convert other audio first: `ffmpeg -i in.mp3 -ar 16000 -ac 1 clip.wav`.
Loading takes about 5 s (the model is converted for the GPU each time it starts).

Check the GPU against onnxruntime step by step, with timings:

```sh
venv/bin/python a4a_gpu.py --steps 6 [--profile] [--wav clip.wav]
```

In your own program, after loading FUTO's driver (`onnx/asr4all_onnx.py`):

```python
import a4a_gpu
engine = asr4all_onnx.load_engine(a4a_gpu.ONNX_DIR)
engine.set_tier("stream_c16r4")       # the only tier compiled for the GPU
a4a_gpu.use_gpu(engine)               # from now on the streaming steps run on the GPU
text = engine.decode_offline(pcm)     # or engine.decode_stream(chunk) ... finish() for live audio
```

Only the streaming acoustic model moves to the GPU. Punctuation and capitals (FUTO's PCEC model) and the
decoder stay on the CPU. They are small.

Tip for live dictation: let the GPU follow the speech, but run the last `finish()` on the CPU (swap
`engine.model.run_stream` back to the original for that call). Text appears 0.15–0.45 s sooner after
you stop talking, at the cost of a short CPU burst at each pause.

## How it works

`a4a_gpu.py` reads the ONNX graph once and turns it into a list of about 550 GPU jobs. `libygpu.so`
(`ygpu.c`) records them into one Vulkan command buffer and replays it for every 0.64 s step.

- Reshape / Slice / Transpose / Split cost nothing: they become views (offset + strides) into one buffer.
- Matrix work (Gemm, MatMul, 1x1 Conv, the spectrogram framing) uses one tuned shader: 12 outputs x 4 frames
  per thread, vec4 loads. BatchNorm is folded into the weights; activations run inside the same job.
- Depthwise convolutions use a sliding window over the input. The attention's "15 shifted windows" chain is one job.
- The two state caches (about 9 MB) stay on the GPU between steps, in a layout chosen so that reading and
  writing them are contiguous copies.
- 256 threads per workgroup throughout. On this GPU that was worth more than any shader rewrite.
- No fp16 or int8 on this GPU, so everything is f32.

Most of each step (~75%) is the matrix work, already close to what this GPU can do. Things tried that did
not help: other tile shapes, uniform weight reads (5–8x slower), elementwise vec4 stores (<1%).

## License

The code here (Python, C, shaders) is MIT, see `LICENSE`. The model is FUTO's and is not included. See
`NOTICE.md` for its license and the uses it forbids.
