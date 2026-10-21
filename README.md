# asr4all-m speech-to-text on the Raspberry Pi 4 / 400 GPU

Runs the streaming acoustic model of FUTO's [asr4all-m](https://huggingface.co/futo-org/asr4all-m)
speech-to-text model on the Pi's own GPU (VideoCore VI, V3D 4.2) through Vulkan, with hand-written compute
shaders. The words come out the same as the normal CPU version, but the CPU is left almost idle.

The GPU is **not faster** than the CPU here. What you get is a free CPU: dictation keeps up with speech
while the CPU stays available for everything else.

## Results (Pi 400)

| | CPU (onnxruntime, 3 threads) | GPU, default 500 MHz | GPU at 600 MHz |
|---|---|---|---|
| Text on 31 real dictation clips (208 s of speech) | — | identical to the CPU on 31/31 | identical on 31/31 |
| Time for those 208 s of speech | ~120 s | ~162 s (0.78x real time) | ~151 s (0.73x real time) |
| CPU time used for them | ~280 s | ~15 s | ~16 s |
| CPU while you talk (live dictation, 2 threads) | ~1 core | ~0.15 core | ~0.15 core |
| One model step (0.64 s of audio) | 359 ms | ~456 ms | ~422 ms |

Accuracy check: log-probabilities within ~3e-5 of onnxruntime, the same best token on every frame.
Raising the GPU clock to 600 MHz (`v3d_freq=600` in /boot/config.txt, then reboot) gives about 7% more speed
and the same text; 550 MHz gave nothing measurable.

## What you need

- A Raspberry Pi 4 or Pi 400. The shaders are tuned for its GPU (V3D 4.2). Pi 5 has a different GPU: untested.
- A 64-bit OS with Mesa's `v3dv` Vulkan driver. **Raspberry Pi OS (64-bit, Bookworm) ships it.**
- About 1.3 GB of free RAM: that is the peak while the model is converted at startup; it then holds about
  0.6 GB (the CPU-only version peaks at about 0.4 GB). With the c32r8 tail tier (below) the peak is about
  2.2 GB and it holds about 1.0 GB. (After setup the host copies of the weights, already on the GPU, are
  freed; `A4A_KEEP_BUILD=1` keeps them for debugging.) A 4 GB Pi is fine; a 2 GB one is too small for the GPU version.

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

Tip for live dictation: let the GPU follow the speech, but run the last `finish()` on the CPU. Text
appears 0.15–0.45 s sooner after you stop talking, at the cost of a short CPU burst at each pause.
While the GPU runs, the engine's caches live on the GPU, so read them back before the CPU takes over:

```python
cpu_run = engine.model.run_stream             # before use_gpu
g = a4a_gpu.use_gpu(engine)
gpu_run = engine.model.run_stream
...
for k in ("filter_cache", "attn_cache"):      # GPU caches -> numpy arrays for the CPU model
    if not isinstance(getattr(engine, k), np.ndarray):
        setattr(engine, k, g.cache(k))
engine.model.run_stream = cpu_run
try:
    engine.finish()
    text = engine.decode_pcec(decode_tail=True)
finally:
    engine.model.run_stream = gpu_run         # next step loads the numpy caches back onto the GPU
```

All-GPU alternative: `use_gpu(engine, tail="stream_c32r8")` also puts the c32r8 tier on the GPU, in the same
context (its own weights, the shared cache layout, the state copied across in ~16 ms). Finish the phrase with one
c32r8 step for the last 17-32 frames instead of two c16r4 steps: the end of a phrase is ~80 ms sooner
(893 vs 972 ms median on 31 clips at 600 MHz); text identical on 30/31 (one mumbled word changed). Costs
~0.9 GB more RAM at startup (peak ~2.2 GB instead of ~1.3 GB; ~1.0 GB held instead of ~0.6 GB) and a longer
first start.

```python
a4a_gpu.use_gpu(engine, tail="stream_c32r8")
tail = next(t for t in engine.tiers if t["name"] == "stream_c32r8")
...                                           # at the end of a phrase, instead of engine.finish():
t16 = engine.tiers[engine.idx]
while engine.next_frame * engine.spf < len(engine.pcm):
    left = -(-len(engine.pcm) // engine.spf) - engine.next_frame
    engine._step(tail if 16 < left <= 32 else t16)
engine.pcec.flush()
engine.pcec.drain()
text = engine.decode_pcec(decode_tail=True)
```

One `A4AGPU` per process: `libygpu.so` keeps one Vulkan context (several tiers can share it, as above), and a second one is refused.

## How it works

`a4a_gpu.py` reads the ONNX graph once and turns it into a list of about 420 GPU jobs. `libygpu.so`
(`ygpu.c`) records them into a Vulkan command buffer and replays it for every 0.64 s step.

- Reshape / Slice / Transpose / Split cost nothing: they become views (offset + strides) into one buffer.
- Matrix work (Gemm, MatMul, 1x1 Conv, the spectrogram framing) uses one tuned shader: 12 outputs x 4 frames
  per thread, vec4 loads. BatchNorm is folded into the weights; activations run inside the same job.
- Depthwise convolutions use a sliding window over the input. The attention's "15 shifted windows" chain is one job.
- The two state caches (about 9 MB) stay on the GPU between steps, in a layout chosen so that reading and
  writing them are contiguous copies.
- An Add or Div right after a matrix job or depthwise conv is done by that job as it stores its result, and a
  BatchNorm of a matrix result is written as a second output (`A4A_FUSE=0` turns this off). Same float
  operations in the same order, so the output is bit-identical; 551 -> 466 jobs, about 4% faster.
- Each depthwise conv's history buffer is a pair that swaps every step, with two recorded command buffers
  (`yg_slot`), so the history never has to be copied in front of the new frames (`A4A_RING=0` = off).
  Another ~2%. `cache()` / `load_cache()` always read and write the current buffer.
- Attention: the two batched matrix products load vec4s (scores along the output row and column, context along
  the shared dimension; `A4A_BMMV=0` = scalar loads), and the scores' scale + mask is done inside the softmax
  that reads them (`A4A_SMFUSE=0` = separate job). About 7 ms (1.4%) per step. The softmax merge is
  bit-identical; the vec4 context product sums in a different order (differences ~1e-5, same text).
- The ~63 cache writes at the end of every step (each history and the attention cache) run as one job that works
  through a table of copies, instead of 63 small jobs that each cost ~0.08 ms just to start. 412 -> 350 jobs,
  about 1.5% faster, bit-identical (`A4A_COPYN=0` = separate jobs).
- The matrix and depthwise-conv shaders get each job's fixed sizes, strides and operation baked in when the
  pipeline is created (Vulkan specialization constants; buffer addresses stay as push constants), so the
  compiler can simplify the indexing and loops. 45 -> 61 pipelines, about 3.5% faster, bit-identical
  (`A4A_SPECIALIZE=0` = generic shaders). The first start after building compiles them (~30 s); later
  starts use the driver's shader cache.
- 256 threads per workgroup throughout. On this GPU that was worth more than any shader rewrite.
- No fp16 or int8 on this GPU, so everything is f32.

Most of each step (~82%) is the matrix work, already close to what this GPU can do. Things tried that did
not help: a softmax that computes each exponential once, other tile shapes, uniform weight reads (5–8x slower), elementwise vec4 stores (<1%), fewer
pipeline barriers, folding the attention's Add+Mul into the matrix job, splitting the narrow matrix jobs
across more threads (split-K) or giving them smaller tiles.

## License

The code here (Python, C, shaders) is MIT, see `LICENSE`. The model is FUTO's and is not included. See
`NOTICE.md` for its license and the uses it forbids.
