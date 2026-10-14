# Model notice

This repository contains only code. It does **not** include any model weights.

It runs the **Asr4All Model** (asr4all-m) made by FUTO Holdings, Inc., which you download yourself from
<https://huggingface.co/futo-org/asr4all-m>. Those weights are licensed under FUTO's own
"FUTO Model Weights License (Restricted Use) 1.0" (the `LICENSE` file that comes with the download), not
under this repository's MIT license. Read it before you use the model: it allows commercial and personal use,
but forbids some uses (for example mass surveillance, profiling people from their voice, intelligence
programs, monitoring people in custody, and targeting journalists or activists).

At load time `a4a_gpu.py` rearranges the weights in memory for the GPU (for example it folds BatchNorm into
the layer before it). Nothing modified is written to disk or distributed.

This project is not made, endorsed or supported by FUTO. FUTO's driver script `onnx/asr4all_onnx.py`
(also part of their download) is used as is and is not included here.
