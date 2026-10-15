#!/usr/bin/env python3
"""a4a_gpu.py — run FUTO asr4all-m's streaming acoustic model (stream_c16r4.onnx) on the GPU of a Raspberry Pi 4 / 400
(V3D 4.2, Vulkan).

Walk the ONNX graph once and turn it into a list of GPU jobs for libygpu.so
(one command buffer per step).
  - Reshape / Slice / Transpose / Unsqueeze / Split are views (offset + strides into one arena buffer).
  - Gemm, constant-weight MatMul, Einsum and 1x1 Conv -> gemm12 (12 outputs x 4 frames per thread);
    activation x activation MatMul -> bmm; other Conv -> conv1d; Softmax/LogSoftmax; ReduceMax; everything
    elementwise -> one strided "ew" shader. BatchNorm = one multiply-add job.
  - The two caches (filter_cache 1.73M floats, attn_cache 548k) stay on the GPU between steps: the step ends
    with copy jobs cache_out -> cache_in. A few tiny nodes depend only on two position counters stored in
    attn_cache (an attention mask etc.); they are evaluated in numpy on the CPU each step ("host" nodes).

Use:  python a4a_gpu.py --steps 6 [--profile] [--wav clip.wav]   compare with onnxruntime step by step + time it
"""
import ctypes
import os
import re
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_DIR = os.environ.get("A4A_MODEL_DIR", os.path.join(HERE, "asr4all-m", "onnx"))   # FUTO's onnx/ folder
VK = os.environ.get("A4A_VULKAN_DIR", "")   # optional private Vulkan loader + icd/; empty = the system's
SHADERS = ["gemm12_00", "gemm12_01", "gemm12_02", "gemm12_10", "gemm12_11", "gemm12_12", "dwconv", "dwconv1", "shiftsum", "copy4", "tcopy", "bmm", "conv1d", "softmax", "rmax"] + ["ew%d" % i for i in range(18)] + \
    ["gemm12p_00", "gemm12p_01", "gemm12p_02", "gemm12p_10", "gemm12p_11", "gemm12p_12", "dwconvp", "dwconv1p"]
PIPE = {n: i for i, n in enumerate(SHADERS)}
ALIGN = 16
WBIT = 0x80000000
NONE = 0xFFFFFFFF
EW = dict(copy=0, add=1, sub=2, mul=3, div=4, max=5, sq=6, sigmoid=7, hsig=8, relu=9, clip=10, log=11, fma=12,
          hswish=13, silu=14, fill=15, scale=16, addmul=17)
EPI = dict(none=0, hswish=1, silu=2, relu=3, sigmoid=4)
# Row trim (2026-10-14, exp/rowtrim.py): every step runs on 38 frames but only frames 0-15 leave it. Every attention
# layer looks at all 38 frames, so only the gemms after the LAST attention layer can skip the look-ahead rows. Proven
# by poisoning rows >= 16 of each: log-probs, hidden and both caches stayed bit-identical (4 steps of a real clip).
# A4A_TRIM=0 turns it off.
ROW_TRIM = {"matmul linear_110": 16, "gemm gemm_output_reshape_arg_token_580": 16,
            "gemm gemm_output_reshape_arg_token_586": 16, "gemm gemm_output_reshape_arg_token_592": 16}


def f2u(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def contig_strides(shape):
    s, out = 1, []
    for n in reversed(shape):
        out.append(s)
        s *= n
    return list(reversed(out))


def groups_for(n_threads, local=256):
    g = max(1, (n_threads + local - 1) // local)
    return (g, 1, 1) if g <= 65535 else (65535, (g + 65534) // 65535, 1)


class Root:
    """A block of memory: kind 'arena' (planned), 'w' (weights buffer), 'fixed' (arena, alive the whole step)."""
    def __init__(self, size, name, kind="arena"):
        self.size, self.name, self.kind = size, name, kind
        self.first = self.last = self.offset = None
        self.alias = None                       # kind 'alias': same memory as another root


class View:
    def __init__(self, root, off, shape, strides=None):
        self.root, self.off, self.shape = root, off, list(shape)
        self.strides = list(strides) if strides is not None else contig_strides(self.shape)

    def size(self):
        return int(np.prod(self.shape)) if self.shape else 1

    def contiguous(self):
        return all(n == 1 or s == c for n, s, c in zip(self.shape, self.strides, contig_strides(self.shape)))

    def as4(self):
        """(shape, strides) padded/merged to exactly 4 dims."""
        sh, st = list(self.shape), list(self.strides)
        while len(sh) > 4:                      # merge leading size-1 dims
            i = sh.index(1) if 1 in sh else None
            assert i is not None, ("5-D view", sh)
            del sh[i], st[i]
        return [1] * (4 - len(sh)) + sh, [0] * (4 - len(st)) + st


def coalesce(shape, strides_list):
    """Merge dims that are contiguous for every operand; drop size-1 dims. -> (shape, [strides...])"""
    dims = [(n, [st[i] for st in strides_list]) for i, n in enumerate(shape) if n != 1]
    out = []
    for n, st in dims:
        if out and all(ps == s * n for ps, s in zip(out[-1][1], st)):
            out[-1] = (out[-1][0] * n, st)
        else:
            out.append((n, st))
    if not out:
        out = [(1, [0] * len(strides_list))]
    return [n for n, _ in out], [[st[k] for _, st in out] for k in range(len(strides_list))]


def bcast(v, shape):
    """View broadcast to `shape` (numpy rules): stride 0 on new / size-1 dims."""
    nd = len(shape)
    sh = [1] * (nd - len(v.shape)) + v.shape
    st = [0] * (nd - len(v.strides)) + v.strides
    st = [0 if a == 1 and b != 1 else s for a, b, s in zip(sh, shape, st)]
    return View(v.root, v.off, shape, st)


def reshape_view(v, shape):
    """No-copy reshape if the strides allow it (numpy's rule), else None."""
    old = [(n, s) for n, s in zip(v.shape, v.strides) if n != 1]
    new = list(shape)
    new_st = [0] * len(new)
    oi, ni = 0, 0
    olen, nlen = len(old), len(new)
    while ni < nlen and oi <= olen:
        if new[ni] == 1:
            new_st[ni] = 0
            ni += 1
            continue
        if oi >= olen:
            return None
        # gather a run of old dims and new dims with equal product
        np_, op_ = new[ni], old[oi][0]
        nj, oj = ni + 1, oi + 1
        while np_ != op_:
            if np_ < op_:
                if nj >= nlen:
                    return None
                np_ *= new[nj]
                nj += 1
            else:
                if oj >= olen:
                    return None
                op_ *= old[oj][0]
                oj += 1
        for k in range(oi, oj - 1):             # old run must be contiguous among itself
            if old[k][1] != old[k + 1][1] * old[k + 1][0]:
                return None
        st = old[oj - 1][1]
        for k in range(nj - 1, ni - 1, -1):
            new_st[k] = st if new[k] != 1 else 0
            st *= new[k]
        ni, oi = nj, oj
    if oi != olen:
        return None
    while ni < nlen:
        new_st[ni] = 0
        ni += 1
    return View(v.root, v.off, new, new_st)


# ---------------- numpy ops for constants and host nodes ----------------
def np_op(n, ins, attr):
    t = n.op_type
    if t == "Add": return ins[0] + ins[1]
    if t == "Sub": return ins[0] - ins[1]
    if t == "Mul": return ins[0] * ins[1]
    if t == "Div": return ins[0] // ins[1] if ins[0].dtype.kind in "iu" else ins[0] / ins[1]
    if t == "Max": return np.maximum(ins[0], ins[1])
    if t == "Min": return np.minimum(ins[0], ins[1])
    if t == "Clip":
        x = ins[0]
        if len(ins) > 1 and ins[1] is not None: x = np.maximum(x, ins[1])
        if len(ins) > 2 and ins[2] is not None: x = np.minimum(x, ins[2])
        return x
    if t == "GreaterOrEqual": return ins[0] >= ins[1]
    if t == "Greater": return ins[0] > ins[1]
    if t == "Less": return ins[0] < ins[1]
    if t == "Equal": return ins[0] == ins[1]
    if t == "And": return np.logical_and(ins[0], ins[1])
    if t == "Or": return np.logical_or(ins[0], ins[1])
    if t == "Not": return np.logical_not(ins[0])
    if t == "Where": return np.where(ins[0], ins[1], ins[2])
    if t == "Cast":
        from onnx import helper
        return ins[0].astype(helper.tensor_dtype_to_np_dtype(attr["to"]))
    if t == "Reshape":
        shape = [int(s) for s in ins[1]]
        if not attr.get("allowzero", 0):
            shape = [ins[0].shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return ins[0].reshape(shape)
    if t == "Unsqueeze":
        x = ins[0]
        axes = sorted(int(a) % (x.ndim + len(ins[1])) for a in ins[1])
        for a in axes: x = np.expand_dims(x, a)
        return x
    if t == "Squeeze":
        return np.squeeze(ins[0], axis=tuple(int(a) for a in ins[1])) if len(ins) > 1 else np.squeeze(ins[0])
    if t == "Concat": return np.concatenate(ins, axis=attr["axis"])
    if t == "Slice":
        x = ins[0]
        starts, ends = ins[1], ins[2]
        axes = ins[3] if len(ins) > 3 and ins[3] is not None else range(len(starts))
        steps = ins[4] if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
        sl = [slice(None)] * x.ndim
        for s, e, a, st in zip(starts, ends, axes, steps):
            sl[int(a)] = slice(int(s), int(e), int(st))
        return x[tuple(sl)]
    if t == "Transpose": return np.transpose(ins[0], attr.get("perm"))
    if t == "Range": return np.arange(ins[0], ins[1], ins[2]).astype(ins[0].dtype)
    if t == "Shape": return np.array(ins[0].shape, np.int64)
    if t == "Gather": return np.take(ins[0], ins[1], axis=attr.get("axis", 0))
    if t == "Expand": return ins[0] * np.ones(ins[1].astype(np.int64), ins[0].dtype)
    if t == "Neg": return -ins[0]
    if t == "Floor": return np.floor(ins[0])
    if t == "Identity": return ins[0]
    raise NotImplementedError("numpy op " + t)


class Compiler:
    def __init__(self, path, debug=False):
        import onnx
        from onnx import numpy_helper, shape_inference, helper
        self.debug = debug
        m = shape_inference.infer_shapes(onnx.load(path))
        g = m.graph
        self.model_path = path
        self.shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
                       for v in list(g.value_info) + list(g.output) + list(g.input)}
        self.consts = {t.name: numpy_helper.to_array(t) for t in g.initializer}
        self.nodes = [n for n in g.node]
        self.attrs = {id(n): {a.name: helper.get_attribute_value(a) for a in n.attribute} for n in self.nodes}
        for n in self.nodes:
            if n.op_type == "Constant":
                self.consts[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
        self.inputs = [i.name for i in g.input]
        self.outputs = [o.name for o in g.output]
        self.consumers = {}
        for n in self.nodes:
            for x in n.input:
                self.consumers.setdefault(x, []).append(n)
        self.roots, self.jobs, self.t = [], [], {}
        self.wparts, self.wsize, self.wcache = [], 0, {}
        self.host_nodes, self.host_vals, self.host_up = [], {}, {}
        self.init_data, self.skip, self.producer, self.chain = [], set(), {}, {}
        self.rings, self.swap = [], {}                   # double-buffered conv histories (see _ring_ok)
        self._build()
        self._redirect_gemms()
        self.fused = self._fuse_post() if os.environ.get("A4A_FUSE", "1") != "0" else 0
        self._plan_memory()
        self._check_rings()

    # ---------- helpers ----------
    def _new(self, name, shape, kind="arena", pitch=True):
        """New tensor. Arena tensors with a last dim >= 8 that is not a multiple of 4 get a padded row pitch,
        so rows start 4-aligned (vec4 loads/stores in gemm12)."""
        shape = list(shape)
        st = None
        size = max(1, int(np.prod(shape))) if shape else 1
        if pitch and kind == "arena" and len(shape) >= 2 and shape[-1] >= 8 and shape[-1] % 4:
            pitch = (shape[-1] + 3) // 4 * 4
            st = contig_strides(shape[:-1] + [pitch])
            size = int(np.prod(shape[:-1])) * pitch
        r = Root(size, name, kind)
        if st is not None:
            r.row = (shape[-1], st[-2])                 # (logical row length, padded pitch)
        self.roots.append(r)
        v = View(r, 0, shape, st)
        self.t[name] = v
        return v

    def _w(self, arr):
        arr = np.array(arr, np.float32).ravel()           # own copy
        self.last_w = arr
        off = self.wsize
        size = (arr.size + ALIGN - 1) // ALIGN * ALIGN
        self.wparts.append(arr)
        if size > arr.size:
            self.wparts.append(np.zeros(size - arr.size, np.float32))
        self.wsize += size
        return off

    def _wview(self, name):
        """A small constant as a view into a fixed arena slot (copied in once at start)."""
        if name not in self.wcache:
            arr = np.asarray(self.consts[name], np.float32)
            v = self._new(name + "#const", list(arr.shape), "fixed")
            self.init_data.append((v, arr))
            self.wcache[name] = v
        return self.wcache[name]

    def _get(self, name):
        if name in self.t:
            return self.t[name]
        if name in self.consts:
            return self._wview(name)
        if name in self.host_vals:          # host value needed on the GPU: upload slot, alive the whole step
            shape = list(np.shape(self.host_vals[name]))
            v = self._new(name, shape, "fixed")
            self.host_up[name] = v
            return v
        raise KeyError(name)

    def _emit(self, pipe, u, n_threads, reads, writes, label, groups=None, local=256):
        self.jobs.append(dict(pipe=pipe, u=u, groups=groups or groups_for(n_threads, local),
                              reads=reads, writes=writes, label=label))

    def _ew(self, op, out, a=None, b=None, c=None, f0=0.0, f1=0.0, label=""):
        n0 = len(self.jobs)
        self._ew1(op, out, a, b, c, f0, f1, label)
        for job in self.jobs[n0:]:
            job["ew_args"] = (op, out, a, b, c, f0, f1, label)

    def _ew1(self, op, out, a, b, c, f0, f1, label):
        ins = [bcast(v, out.shape) if v is not None else None for v in (a, b, c)]
        st_list = [out.strides] + [v.strides if v is not None else [0] * len(out.shape) for v in ins]
        order = sorted(range(len(out.shape)), key=lambda i: -out.strides[i])   # walk in output memory order
        st_list = [[st[i] for i in order] for st in st_list]
        sh, sts = coalesce([out.shape[i] for i in order], st_list)
        assert len(sh) <= 4, (out.shape, sh)
        pad = 4 - len(sh)
        sh = [1] * pad + sh
        sts = [[0] * pad + st for st in sts]
        u = list(sh) + [out] + sts[0]
        for v, st in zip(ins, sts[1:]):
            u += ([v] if v is not None else [0]) + st
        if op == "copy" and self._fast_copy(out, a, sh, sts, label):
            return
        u += [EW[op], f2u(f0), f2u(f1), f2u(1.0 / sh[3])]
        reads = [v for v in (a, b, c) if v is not None]
        nx = sh[2] * sh[3]
        assert nx < (1 << 22) and sh[1] <= 65535, sh
        self._emit("ew%d" % EW[op], u, 0, reads, [out], label or op, groups=((nx + 255) // 256, sh[1], 1))

    def _fast_copy(self, out, a, sh, sts, label):
        """vec4 copy when the last dim is contiguous + 4-aligned on both sides; tiled transpose when the two
        sides are contiguous along different dims. False -> generic ew copy."""
        so, sa = sts[0], sts[1]
        aligned = lambda v, st: v.off % 4 == 0 and all(x % 4 == 0 for x in st[:3])
        if sh[3] % 4 == 0 and so[3] == 1 and sa[3] == 1 and aligned(out, so) and aligned(a, sa) \
                and out.root.kind != "fixed" or (sh[3] % 4 == 0 and so[3] == 1 and sa[3] == 1 and aligned(out, so)
                                                 and aligned(a, sa)):
            u = list(sh) + [out] + so + [a] + sa + [0] * 13 + [f2u(1.0 / (sh[3] // 4))]
            nx = sh[2] * (sh[3] // 4)
            self._emit("copy4", u, 0, [a], [out], label + " v4", groups=((nx + 255) // 256, sh[1], 1))
            return True
        if sh[3] % 4 and so[3] == 1 and sa[3] == 1 and aligned(out, so) and aligned(a, sa) \
                and self._pad_room(out, so, sh[3], True) and self._pad_room(a, sa, sh[3], False):
            w = (sh[3] + 3) // 4 * 4                    # copy whole vec4s: the extra columns are row padding
            u = sh[:3] + [w] + [out] + so + [a] + sa + [0] * 13 + [f2u(1.0 / (w // 4))]
            self._emit("copy4", u, 0, [a], [out], label + " v4p", groups=((sh[2] * (w // 4) + 255) // 256, sh[1], 1))
            return True
        if sh[0] == 1 and sh[1] == 1 and sa[3] == 1 and so[2] == 1 and so[3] != 1:
            # out[i][j] with i = dim 2 (out contiguous), j = dim 3 (A contiguous)
            I, J = sh[2], sh[3]
            u = [1, I, J, out, 0, so[2], so[3], a, 0, sa[2], sa[3]]
            self._emit("tcopy", u, 0, [a], [out], label + " T", groups=((J + 15) // 16, (I + 15) // 16, 1))
            return True
        if sh[0] == 1 and sh[1] == 1 and sa[2] == 1 and so[3] == 1 and sa[3] != 1:
            I, J = sh[3], sh[2]                            # swap roles: i = dim 3, j = dim 2
            u = [1, I, J, out, 0, so[3], so[2], a, 0, sa[3], sa[2]]
            self._emit("tcopy", u, 0, [a], [out], label + " T", groups=((J + 15) // 16, (I + 15) // 16, 1))
            return True
        return False

    @staticmethod
    def _pad_room(v, st, n, at_end):
        """Rows of v (last dim n, contiguous) can be widened to a multiple of 4 inside the root's row padding.
        at_end: the row must also end exactly at the logical row end (so only padding is overwritten)."""
        row = getattr(v.root, "row", None)
        if row is None or v.root.kind != "arena":
            return False
        L, P = row
        col = v.off % P
        return all(x % P == 0 for x in st[:3]) and col + (n + 3) // 4 * 4 <= P and (not at_end or col + n == L)

    def _copy(self, src, dst, label="copy"):
        self._ew("copy", dst, src, label=label)

    def _materialize(self, v, name, pitch=True):
        out = self._new(name + "#c", v.shape, pitch=pitch)
        self._copy(v, out, "copy " + name)
        return out

    # ---------- graph walk ----------
    def _host_seed(self, n):
        """Tiny slices of the attention cache (position counters) start the host subgraph."""
        if n.op_type != "Slice" or n.input[0] != "attn_cache":
            return False
        return int(np.prod(self.shapes.get(n.output[0], [10**9]))) <= 4

    def _build(self):
        for name in self.inputs:
            if name.endswith("_pcm"):
                self.pcm_name = name
            self._new(name, self.shapes[name], "fixed")
        self.in_views = {n: self.t[n] for n in self.inputs}
        self._find_sum_chains()
        self._find_cache_trees()
        self.attn_seeds = []
        for idx, n in enumerate(self.nodes):
            if n.op_type == "Constant":
                continue
            ins = list(n.input)
            att = self.attrs[id(n)]
            known = [x == "" or x in self.consts for x in ins]
            if all(known):                                       # constant folding
                vals = [self.consts.get(x) if x else None for x in ins]
                for o, v in zip(n.output, self._np_outputs(n, vals, att)):
                    self.consts[o] = v
                continue
            if self._host_seed(n):
                self.attn_seeds.append(n)
                self.host_nodes.append(n)
                self.host_vals[n.output[0]] = np.zeros(self.shapes[n.output[0]], np.float32)
                continue
            if all(x == "" or x in self.consts or x in self.host_vals for x in ins) and \
                    any(x in self.host_vals for x in ins):
                self.host_nodes.append(n)
                vals = [self.consts.get(x, self.host_vals.get(x)) if x else None for x in ins]
                for o, v in zip(n.output, self._np_outputs(n, vals, att)):
                    self.host_vals[o] = v
                continue
            if id(n) in self.tree_nodes:
                if id(n) in self.cache_tree:                     # cache_out: write every piece into the cache slot
                    self._write_cache(n)
                continue
            if id(n) in self.skip:
                continue
            getattr(self, "_op_" + n.op_type)(n, ins, att)
            if n.op_type in ("Reshape", "Transpose", "Unsqueeze", "Squeeze") and ins[0] in self.chain:
                if len(self.consumers.get(ins[0], [])) == 1:
                    self.chain[n.output[0]] = self.chain[ins[0]]
        # outputs: caches written back into the input slots by a final copy
        self.out_views = {}
        for o in self.outputs:
            cache = "filter_cache" if o.endswith("filter_cache_out") else "attn_cache" if o.endswith("attn_cache_out") else None
            if cache is None:
                self.out_views[o] = self.t[o]
                continue
            if o not in self.t:                                   # written in place by _write_cache
                continue
            v = self.t[o]
            src = self.in_views[cache]
            uses = lambda job, r: any(x.root is r for x in job["reads"] + job["writes"])
            last_read = max((j for j, job in enumerate(self.jobs) if uses(job, src.root)), default=-1)
            writes = [j for j, job in enumerate(self.jobs) if uses(job, v.root)]
            reads_out = [j for j, job in enumerate(self.jobs) if any(x.root is v.root for x in job["reads"])]
            if v.contiguous() and v.off == 0 and v.size() == src.size() and writes and not reads_out \
                    and last_read < min(writes) and v.root.kind == "arena":
                v.root.kind, v.root.alias = "alias", src.root        # concat writes the new cache in place
                self.cache_alias = getattr(self, "cache_alias", []) + [cache]
            else:
                self._copy(v, src, "cache back " + cache)

    # ---------- cache layout ----------
    # filter_cache holds 50 per-layer [C][T] conv histories. They feed channels-last ([T][C]) depthwise-conv
    # buffers, so on the GPU each slot is stored transposed ([T][C]): reading a slot and writing the next one
    # are then contiguous vec4 copies instead of transposes. to_internal / from_internal convert for the CPU.
    def _find_cache_trees(self):
        """The Concat/Reshape/Unsqueeze/Pad nodes that only assemble a cache output. They are not run as jobs:
        _write_cache copies each leaf straight into its place in the cache input slot."""
        self.prod = {o: n for n in self.nodes for o in n.output}
        self.cache_tree, self.tree_nodes = {}, set()
        self.tlayout = {}                                 # cache name -> {flat offset: (C, T)} stored transposed

        def walk(n):
            self.tree_nodes.add(id(n))
            for x in (n.input if n.op_type == "Concat" else n.input[:1]):
                p = self.prod.get(x)
                if p is None or len(self.consumers.get(x, [])) != 1 or x in self.outputs:
                    continue
                if p.op_type in ("Concat", "Reshape", "Unsqueeze", "Squeeze") or \
                        (p.op_type == "Pad" and self._zero_pad(p)):
                    walk(p)
        for o in self.outputs:
            cache = self._cache_of(o)
            if cache:
                self.cache_tree[id(self.prod[o])] = cache
                walk(self.prod[o])

    @staticmethod
    def _cache_of(o):
        return "filter_cache" if o.endswith("filter_cache_out") else "attn_cache" if o.endswith("attn_cache_out") else None

    def _zero_pad(self, p):
        a = self.attrs[id(p)]
        return a.get("mode", b"constant") in (b"constant", "constant") and p.input[1] in self.consts and \
            (len(p.input) < 3 or not p.input[2] or (p.input[2] in self.consts and float(self.consts[p.input[2]]) == 0))

    def _cache_reshape(self, v, shape):
        """reshape_view, plus: a flat view of filter_cache reshaped to [.., C, T] uses the transposed slot layout."""
        r = reshape_view(v, shape)
        fc = self.in_views.get("filter_cache")
        if r is None or fc is None or v.root is not fc.root or len(v.shape) != 1 or len(shape) < 2:
            return r
        C, T = shape[-2], shape[-1]
        n = C * T
        assert v.size() % n == 0, ("filter_cache slot not whole", v.off, shape)
        slots = self.tlayout.setdefault("filter_cache", {})
        for k in range(v.size() // n):
            if slots.setdefault(v.off + k * n, (C, T)) != (C, T):
                raise RuntimeError("filter_cache slot read with two shapes")
        r.strides[-2:] = [1, C]
        return r

    def _write_cache(self, n):
        cache = self.cache_tree[id(n)]
        dest = self.in_views[cache]
        self.first_cache_write = getattr(self, "first_cache_write", {})
        self.first_cache_write[cache] = len(self.jobs)
        self._write_into(n.output[0], dest)
        # every job that reads the cache input must come before the writes (they overwrite it in place)
        assert not any(x.root is dest.root for job in self.jobs[self.first_cache_write[cache]:]
                       for x in job["reads"]), "cache read after its in-place write"

    def _write_into(self, name, dest):
        p = self.prod.get(name)
        if p is None or id(p) not in self.tree_nodes or name in self.host_vals or name in self.consts or name in self.t:
            self._copy(self._get(name), self._ring_dest(dest), "cache write " + name)
            return
        shape_in = lambda x: list(self.shapes[x]) if x in self.shapes else None
        if p.op_type == "Concat":
            ax = self.attrs[id(p)]["axis"] % len(dest.shape)
            pos = 0
            for x in p.input:
                k = shape_in(x)[ax]
                sh = list(dest.shape)
                sh[ax] = k
                self._write_into(x, View(dest.root, dest.off + pos * dest.strides[ax], sh, dest.strides))
                pos += k
            assert pos == dest.shape[ax], (name, pos, dest.shape)
        elif p.op_type in ("Reshape", "Unsqueeze", "Squeeze"):
            v = self._cache_reshape(dest, shape_in(p.input[0]))
            assert v is not None, ("cache write: no view for", name, dest.shape, dest.strides)
            self._write_into(p.input[0], v)
        elif p.op_type == "Pad":                         # padding stays zero: never written, loaded as zeros
            pads = [int(x) for x in self.consts[p.input[1]]]
            sh = shape_in(p.input[0])
            nd = len(sh)
            assert all(x >= 0 for x in pads)
            self._write_into(p.input[0], View(dest.root, dest.off + sum(pads[i] * dest.strides[i] for i in range(nd)),
                                              sh, dest.strides))
        else:
            raise AssertionError(p.op_type)

    def to_internal(self, cache, arr):
        out = np.array(arr, np.float32).ravel()
        for off, (C, T) in self.tlayout.get(cache, {}).items():
            out[off:off + C * T] = out[off:off + C * T].reshape(C, T).T.ravel()
        return out

    def from_internal(self, cache, arr):
        out = np.array(arr, np.float32).ravel()
        for off, (C, T) in self.tlayout.get(cache, {}).items():
            out[off:off + C * T] = out[off:off + C * T].reshape(T, C).T.ravel()
        return out

    # ---------- gemm -> channels-last concat ----------
    def _redirect_gemms(self):
        """A gemm output ([N][M] channels-first) that is copied whole (transposed) into a channels-last concat
        buffer: the gemm writes into the buffer itself (vec4 along n) and the copy is dropped. Its other readers
        (elementwise jobs only) are re-emitted with views into the buffer."""
        outs = {id(v.root) for v in self.out_views.values()}
        drop = set()
        for ci, cj in enumerate(self.jobs):
            if cj["pipe"] != "tcopy" or not cj["label"].startswith("concat"):
                continue
            s, d = cj["reads"][0], cj["writes"][0]
            R = s.root
            j = self.producer.get(id(R))
            if j is None or not self.jobs[j]["pipe"].startswith("gemm") or id(R) in outs or R.kind != "arena":
                continue
            g = self.jobs[j]
            gout, M, N = g["u"][9], g["u"][0], g["u"][1]
            if gout.root is not R or gout.off != 0 or gout.strides[0] != 1:
                continue
            P = gout.strides[1]
            dims = {st: (n, ds) for n, st, ds in zip(s.shape, s.strides, d.strides) if n != 1}
            if s.off != 0 or set(dims) != {1, P} or dims[1][0] != M or dims[P][0] != N:
                continue
            dm, dn = dims[1][1], dims[P][1]

            def remap(v):
                n0, m0 = divmod(v.off, P)
                st = [0 if n == 1 else dm if x == 1 else dn if x == P else None for n, x in zip(v.shape, v.strides)]
                return None if None in st else View(d.root, d.off + n0 * dn + m0 * dm, v.shape, st)
            users = [k for k, jb in enumerate(self.jobs) if k not in (j, ci) and k not in drop
                     and any(v.root is R for v in jb["reads"] + jb["writes"])]
            if any("ew_args" not in self.jobs[k] or any(v.root is R for v in self.jobs[k]["writes"]) for k in users):
                continue
            new_args = {}
            for k in users:
                args = list(self.jobs[k]["ew_args"])
                for i in (1, 2, 3, 4):
                    if isinstance(args[i], View) and args[i].root is R:
                        args[i] = remap(args[i])
                if any(x is None for x in args[1:5] if not isinstance(x, (type(None), View))) or \
                        any(args[i] is None and self.jobs[k]["ew_args"][i] is not None for i in (1, 2, 3, 4)):
                    break
                new_args[k] = args
            else:
                o = View(d.root, d.off, [M, N], [dm, dn])
                ov = 2 if dn == 1 and dm % 4 == 0 and o.off % 4 == 0 and N % 4 == 0 else 0
                g["u"][9:12] = [o, dm, dn]
                g["pipe"] = g["pipe"][:-1] + str(ov)
                g["writes"] = [o]
                g["label"] += " ->" + d.root.name
                for k, args in new_args.items():
                    n0 = len(self.jobs)
                    self._ew(*args)
                    assert len(self.jobs) == n0 + 1
                    self.jobs[k] = self.jobs.pop()
                drop.add(ci)
        self.jobs = [jb for k, jb in enumerate(self.jobs) if k not in drop]
        self.redirected = len(drop)

    # ---------- Add / Div folded into the gemm / depthwise conv that made one input (2026-10-14) ----------
    # out = X + R (or X / R for convs) where X is read by nothing else: the producer writes epi(...) + R straight
    # into the Add's output and the elementwise job is dropped. Same float operations in the same order, so the
    # result is bit-identical (checked: log-probs, hidden and both caches over 10 real-speech steps).
    def _fuse_post(self):
        outs = {id(v.root) for v in self.out_views.values()}
        n = 0
        while True:
            touch = {}
            for k, jb in enumerate(self.jobs):
                for v in jb["reads"]:
                    touch.setdefault(id(v.root), []).append((k, "r"))
                for v in jb["writes"]:
                    touch.setdefault(id(v.root), []).append((k, "w"))
            for k, jb in enumerate(self.jobs):
                if jb["pipe"] not in ("ew1", "ew4", "ew17", "ew12") or "ew_args" not in jb:
                    continue
                op, O, a, b, c = jb["ew_args"][:5]
                if op == "addmul":                        # tried 2026-10-14: no gain (B is channels-last)
                    ok = False
                elif op == "fma":                         # a * b + c with b, c per channel: BatchNorm
                    ok = self._fuse_bn(k, a, b, c, O, touch, outs)
                else:
                    pairs = [(a, b), (b, a)] if op == "add" else [(a, b)]
                    ok = any(self._fuse_one(k, X, R, O, op, touch, outs) for X, R in pairs)
                if ok:
                    del self.jobs[k]
                    n += 1
                    break
            else:
                return n

    def _gemm_coords(self, g, X):
        """axis of X that is gemm row m / column n (None if X is not exactly the gemm's output)."""
        M, N, sm, sn = g["u"][0], g["u"][1], g["u"][10], g["u"][11]
        if g["writes"][0].off != X.off or sm == sn:
            return None
        dims = {}
        for d, (sz, st) in enumerate(zip(X.shape, X.strides)):
            if sz == 1:
                continue
            key = "m" if (st, sz) == (sm, M) else "n" if (st, sz) == (sn, N) else None
            if key is None or key in dims:
                return None
            dims[key] = d
        return dims if set(dims) == {"m", "n"} else None

    def _gemm_post_init(self, g):
        if re.fullmatch(r"gemm12_\d\d", g["pipe"]):
            g["u"] = list(g["u"][:13]) + [0, 0, 0, NONE, 0, 0, 0, 0]
            g["pipe"] = "gemm12p_" + g["pipe"][7:]

    def _fuse_bn(self, k, X, S, T, O, touch, outs):
        """O = X * S + T (S, T per column) where X is a gemm output (OV4 1): the gemm also writes O."""
        if X is None or S is None or T is None or O.root.kind != "arena" or list(X.shape) != list(O.shape):
            return False
        ws = [i for i, m in touch.get(id(X.root), []) if m == "w"]
        if len(ws) != 1 or ws[0] >= k:
            return False
        j = ws[0]
        g = self.jobs[j]
        if not re.fullmatch(r"gemm12p?_\d1", g["pipe"]) or len(g["writes"]) != 1 or \
                (len(g["u"]) > 16 and not (isinstance(g["u"][16], int) and g["u"][16] == NONE)):
            return False
        dims = self._gemm_coords(g, X)
        if dims is None or any(j <= i < k for i, m in touch.get(id(X.root), []) if m == "w" and i != j):
            return False
        if any(j < i < k for i, _ in touch.get(id(O.root), [])) or O.root is X.root:
            return False
        Sb, Tb = bcast(S, O.shape), bcast(T, O.shape)
        for V in (Sb, Tb):
            if V.strides[dims["m"]] != 0 or any(i >= j for i, m in touch.get(id(V.root), []) if m == "w"):
                return False
        W = g["writes"][0]
        on, q4 = W.strides[1], (g["u"][0] + 3) // 4 * 4
        if O.strides[dims["m"]] != W.strides[0] or O.strides[dims["n"]] != on or W.strides[0] != 1 or \
                O.off % 4 or (O.off - W.off) % 4 or O.off % on + q4 > on:
            return False
        self._gemm_post_init(g)
        g["u"][16:21] = [View(O.root, O.off, list(W.shape), list(W.strides)), Sb, Tb,
                         Sb.strides[dims["n"]], Tb.strides[dims["n"]]]
        g["u"][17] = View(Sb.root, Sb.off, [1], [0])
        g["u"][18] = View(Tb.root, Tb.off, [1], [0])
        g["reads"] = list(g["reads"]) + [Sb, Tb]
        g["writes"] = list(g["writes"]) + [g["u"][16]]
        g["label"] += " +bn"
        return True

    def _fuse_one(self, k, X, R, O, op, touch, outs):
        if X is None or R is None or X.root is R.root or R.root is O.root or X.root is O.root:
            return False
        if X.root.kind != "arena" or O.root.kind != "arena" or id(X.root) in outs or list(X.shape) != list(O.shape):
            return False
        tx = touch.get(id(X.root), [])
        js = [i for i, m in tx if m == "w"]
        if len(js) != 1 or {i for i, _ in tx} != {js[0], k}:
            return False
        j = js[0]
        g = self.jobs[j]
        kind = "gemm" if re.fullmatch(r"gemm12_\d\d", g["pipe"]) else "dw" if g["pipe"] in ("dwconv", "dwconv1") else None
        if kind is None or (op == "div" and kind != "dw") or len(g["writes"]) != 1:
            return False
        if any(i >= j for i, m in touch.get(id(R.root), []) if m == "w" and i != k):
            return False                                  # R must be final before the producer runs
        if any(j < i < k for i, _ in touch.get(id(O.root), [])):
            return False                                  # nothing between may see the Add output early
        W = g["writes"][0]
        Rb = bcast(R, O.shape)
        if kind == "gemm":
            M, N, sm, sn = g["u"][0], g["u"][1], g["u"][10], g["u"][11]
        else:
            N, M, sn, sm = g["u"][0], g["u"][1], g["u"][12], g["u"][13]   # dw: (C, T) as (n, m)
        if W.off != X.off or sm == sn:
            return False
        dims = {}                                         # "m" / "n" -> axis of X
        for d, (sz, st) in enumerate(zip(X.shape, X.strides)):
            if sz == 1:
                continue
            key = "m" if (st, sz) == (sm, M) else "n" if (st, sz) == (sn, N) else None
            if key is None or key in dims:
                return False
            dims[key] = d
        if set(dims) != {"m", "n"}:
            return False
        oS = {key: O.strides[d] for key, d in dims.items()}
        rS = {key: Rb.strides[d] for key, d in dims.items()}
        if kind == "gemm":
            if rS != oS or op != "add":
                return False
            om, on = oS["m"], oS["n"]
            q4 = (M + 3) // 4 * 4
            ops = [Rb]
            ov = 1 if om == 1 and on % 4 == 0 and O.off % 4 == 0 and O.off % on + q4 <= on and \
                all(V.off % 4 == 0 and V.off % on + q4 <= on for V in ops) else \
                2 if on == 1 and om % 4 == 0 and O.off % 4 == 0 and N % 4 == 0 and all(V.off % 4 == 0 for V in ops) else 0
            o = View(O.root, O.off, [M, N], [om, on])
            av = g["pipe"][7]
            self._gemm_post_init(g)
            g["u"][9:12] = [o, om, on]
            g["u"][13:15] = [View(Rb.root, Rb.off, [M, N], [om, on]), 1]
            g["pipe"] = "gemm12p_" + av + str(ov)
        else:
            oc, ot = oS["n"], oS["m"]
            cl = int(oc == 1 and ot % 4 == 0 and O.off % 4 == 0)
            o = View(O.root, O.off, [1, N, M], [0, oc, ot])
            g["u"][11:14] = [o, oc, ot]
            g["u"][15] = cl
            g["u"] = list(g["u"][:16]) + [0, 0, 0, View(Rb.root, Rb.off, [N, M], [rS["n"], rS["m"]]),
                                          rS["n"], rS["m"], 1 if op == "add" else 2]
            g["pipe"] += "p"
        g["reads"] = list(g["reads"]) + [Rb]
        g["writes"] = [o]
        g["label"] += " +" + op
        return True

    # ---------- sum-of-products chains (attention shift-sum) ----------
    def _find_sum_chains(self):
        """Add chains whose terms are all single-use Muls -> one job at the last Add (_op_Add)."""
        prod = {o: n for n in self.nodes for o in n.output}
        self.sum_chains = {}
        for n in self.nodes:
            if n.op_type != "Add":
                continue
            cs = self.consumers.get(n.output[0], [])
            if len(cs) == 1 and cs[0].op_type == "Add":
                continue                                   # not the last Add of a chain
            terms, inner, ok = [], [], True

            def walk(x):
                nonlocal ok
                q = prod.get(x)
                if q is None or len(self.consumers.get(x, [])) != 1:
                    ok = False
                elif q.op_type == "Add":
                    inner.append(q)
                    walk(q.input[0])
                    walk(q.input[1])
                elif q.op_type == "Mul":
                    terms.append(q)
                else:
                    ok = False
            walk(n.input[0])
            walk(n.input[1])
            if ok and len(terms) >= 3:
                self.sum_chains[id(n)] = terms
                for q in terms + inner:
                    self.skip.add(id(q))

    def _shiftsum(self, n, terms):
        """sum_k P_k * V_k with P_k [1,H,1,T] / V_k [1,H,D,T] views at equally spaced offsets; else None."""
        pairs = []
        for q in terms:
            a, b = self._get(q.input[0]), self._get(q.input[1])
            if len(a.shape) != 4 or len(b.shape) != 4:
                return None
            P, V = (a, b) if a.shape[2] == 1 else (b, a)
            if P.shape[2] != 1:
                return None
            pairs.append((P, V))
        pairs.sort(key=lambda pv: pv[0].off)
        P0, V0 = pairs[0]
        H, D, T = V0.shape[1], V0.shape[2], V0.shape[3]
        if P0.shape != [1, H, 1, T]:
            return None
        dP = pairs[1][0].off - P0.off
        dV = pairs[1][1].off - V0.off
        for k, (P, V) in enumerate(pairs):
            if P.root is not P0.root or V.root is not V0.root or P.strides != P0.strides or V.strides != V0.strides \
                    or P.shape != P0.shape or V.shape != V0.shape or P.off != P0.off + k * dP or V.off != V0.off + k * dV:
                return None
        out = self._new(n.output[0], [1, H, D, T])
        u = [H, D, T, len(pairs), P0, P0.strides[1], P0.strides[3], dP, V0, V0.strides[1], V0.strides[2],
             V0.strides[3], dV, out, out.strides[1], out.strides[2], out.strides[3]]
        self._emit("shiftsum", u, H * D * T, [p for pv in pairs for p in pv], [out], "shiftsum " + n.output[0])
        return out

    def _op_Add(self, n, ins, att):
        terms = self.sum_chains.get(id(n))
        if terms is not None:
            if self._shiftsum(n, terms) is None:          # not the expected shape: plain jobs instead
                acc = None
                for q in terms:
                    a, b = self._get(q.input[0]), self._get(q.input[1])
                    shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
                    out = self._new(n.output[0] + "#acc%d" % id(q), shape)
                    if acc is None:
                        self._ew("mul", out, a, b)
                    else:
                        self._ew("addmul", out, acc, a, b)
                    acc = out
                self.t[n.output[0]] = acc
            return
        self._binary(n, ins, "add")

    def _np_outputs(self, n, vals, att):
        if n.op_type == "Split":
            x = vals[0]
            ax = att.get("axis", 0)
            if len(vals) > 1 and vals[1] is not None:
                parts = np.cumsum(vals[1])[:-1]
            else:
                parts = len(n.output)
            return np.split(x, parts, axis=ax)
        return [np_op(n, vals, att)]

    # ---------- view ops ----------
    def _op_Reshape(self, n, ins, att):
        v = self._get(ins[0])
        shape = [int(s) for s in self.consts[ins[1]]]
        if not att.get("allowzero", 0):
            shape = [v.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        if -1 in shape:
            k = shape.index(-1)
            shape[k] = v.size() // int(np.prod([s for s in shape if s != -1]))
        r = self._cache_reshape(v, shape)
        if r is None:
            r = reshape_view(self._materialize(v, n.output[0]), shape)
        if r is None:
            r = reshape_view(self._materialize(v, n.output[0], pitch=False), shape)
        self.t[n.output[0]] = r

    def _op_Unsqueeze(self, n, ins, att):
        v = self._get(ins[0])
        axes = sorted(int(a) % (len(v.shape) + len(self.consts[ins[1]])) for a in self.consts[ins[1]])
        sh, st = list(v.shape), list(v.strides)
        for a in axes:
            sh.insert(a, 1)
            st.insert(a, 0)
        self.t[n.output[0]] = View(v.root, v.off, sh, st)

    def _op_Squeeze(self, n, ins, att):
        v = self._get(ins[0])
        axes = [int(a) % len(v.shape) for a in self.consts[ins[1]]] if len(ins) > 1 else \
            [i for i, s in enumerate(v.shape) if s == 1]
        keep = [i for i in range(len(v.shape)) if i not in axes]
        self.t[n.output[0]] = View(v.root, v.off, [v.shape[i] for i in keep], [v.strides[i] for i in keep])

    def _op_Transpose(self, n, ins, att):
        v = self._get(ins[0])
        perm = att.get("perm", list(range(len(v.shape)))[::-1])
        self.t[n.output[0]] = View(v.root, v.off, [v.shape[p] for p in perm], [v.strides[p] for p in perm])

    def _op_Slice(self, n, ins, att):
        v = self._get(ins[0])
        starts, ends = self.consts[ins[1]], self.consts[ins[2]]
        axes = self.consts[ins[3]] if len(ins) > 3 and ins[3] else range(len(starts))
        steps = self.consts[ins[4]] if len(ins) > 4 and ins[4] else [1] * len(starts)
        sh, st, off = list(v.shape), list(v.strides), v.off
        for s, e, a, k in zip(starts, ends, axes, steps):
            a = int(a) % len(sh)
            s, e, k = slice(int(s), int(e), int(k)).indices(sh[a])
            cnt = len(range(s, e, k))
            off += s * st[a]
            st[a] *= k
            sh[a] = cnt
        self.t[n.output[0]] = View(v.root, off, sh, st)

    def _op_Split(self, n, ins, att):
        v = self._get(ins[0])
        ax = att.get("axis", 0) % len(v.shape)
        if len(ins) > 1 and ins[1]:
            sizes = [int(s) for s in self.consts[ins[1]]]
        else:
            k = len(n.output)
            sizes = [v.shape[ax] // k] * k
        pos = 0
        for o, s in zip(n.output, sizes):
            sh = list(v.shape)
            sh[ax] = s
            self.t[o] = View(v.root, v.off + pos * v.strides[ax], sh, v.strides)
            pos += s

    def _dw_consumer(self, name):
        """True if some consumer of `name` is a depthwise Conv (it then wants channels-last [T][C] storage)."""
        for c in self.consumers.get(name, []):
            if c.op_type == "Conv" and c.input[0] == name:
                at = self.attrs[id(c)]
                W = self.consts.get(c.input[1])
                if W is not None and at.get("group", 1) == W.shape[0] and W.shape[1] == 1 and W.shape[0] % 4 == 0:
                    return True
        return False

    def _new_cl(self, name, shape):
        """[1, C, T] tensor stored channels-last ([T][C] memory)."""
        st = self._new(name + "#cl", [shape[0], shape[2], shape[1]])
        v = View(st.root, 0, list(shape), [st.strides[0], st.strides[2], st.strides[1]])
        self.t[name] = v
        return v

    def _op_Concat(self, n, ins, att):
        vs = [self._get(x) for x in ins]
        ax = att["axis"] % len(vs[0].shape)
        shape = list(vs[0].shape)
        shape[ax] = sum(v.shape[ax] for v in vs)
        ring = self._ring_ok(n, vs, ax, shape)
        if ring:
            out = self._new_ring(n.output[0], shape, vs[0])
        elif len(shape) == 3 and shape[0] == 1 and ax == 2 and self._dw_consumer(n.output[0]):
            out = self._new_cl(n.output[0], shape)
        else:
            flat = all(c.op_type == "Reshape" for c in self.consumers.get(n.output[0], []))
            out = self._new(n.output[0], shape, pitch=not flat)
        pos = 0
        for k, v in enumerate(vs):
            sh = list(shape)
            sh[ax] = v.shape[ax]
            if not (ring and k == 0):                    # ring: the history is already in rows 0..hist-1
                self._copy(v, View(out.root, pos * out.strides[ax], sh, out.strides), "concat " + n.output[0])
            pos += v.shape[ax]

    # ---------- double-buffered depthwise-conv histories (2026-10-14) ----------
    # concat([filter_cache slot (hist frames), new frames]) -> depthwise conv, and the next history is a Slice of
    # that concat. The concat buffer becomes two persistent buffers A/B: this step's history already sits in
    # A rows 0..hist-1 (no copy in), the next history is written to B rows 0..hist-1, and the second recorded
    # plan is the same with A and B swapped. Rows are [T][C] like the transposed filter_cache slots.
    def _check_rings(self):
        fc = self.in_views.get("filter_cache")
        for r in self.rings:
            if not r.get("written"):
                raise RuntimeError("ring history never saved: " + r["name"])
            for jb in self.jobs:
                for v in jb["reads"]:
                    if v.root is fc.root:
                        lo = v.off + sum(min(0, (n - 1) * s) for n, s in zip(v.shape, v.strides))
                        hi = v.off + sum(max(0, (n - 1) * s) for n, s in zip(v.shape, v.strides))
                        if lo < r["off"] + r["n"] and r["off"] <= hi:
                            raise RuntimeError("ring slot still read from filter_cache: " + jb["label"])

    def _ring_ok(self, n, vs, ax, shape):
        fc = self.in_views.get("filter_cache")
        if os.environ.get("A4A_RING", "1") == "0" or fc is None or len(vs) != 2 or ax != 2 or len(shape) != 3:
            return False
        v, C = vs[0], shape[1]
        if v.root is not fc.root or v.strides[1] != 1 or v.strides[2] != C or v.off % 4 or C % 4 or shape[0] != 1:
            return False
        if not self._dw_consumer(n.output[0]):
            return False
        for c in self.consumers.get(n.output[0], []):
            if c.op_type == "Slice":
                if not all(id(x) in self.tree_nodes for x in self.consumers.get(c.output[0], [])):
                    return False
            elif not (c.op_type == "Conv" and c.input[0] == n.output[0]):
                return False
        return True

    def _new_ring(self, name, shape, hist):
        T, C = shape[2], shape[1]
        A = self._new(name + "#ringA", [1, T, C], "fixed")
        B = self._new(name + "#ringB", [1, T, C], "fixed")
        v = View(A.root, 0, list(shape), [T * C, 1, C])
        self.t[name] = v
        self.rings.append(dict(A=A.root, B=B.root, off=hist.off, n=C * hist.shape[2], name=name))
        self.swap[id(A.root)], self.swap[id(B.root)] = B.root, A.root
        return v

    def _ring_dest(self, dest):
        """A cache-out write into a ring slot of filter_cache goes to the B buffer instead."""
        fc = self.in_views.get("filter_cache")
        if fc is None or dest.root is not fc.root:
            return dest
        lo = dest.off + sum(min(0, (n - 1) * s) for n, s in zip(dest.shape, dest.strides))
        hi = dest.off + sum(max(0, (n - 1) * s) for n, s in zip(dest.shape, dest.strides))
        for r in self.rings:
            if r["off"] <= lo and hi < r["off"] + r["n"]:
                r["written"] = True
                return View(r["B"], dest.off - r["off"], dest.shape, dest.strides)
            if lo < r["off"] + r["n"] and r["off"] <= hi:
                raise RuntimeError("cache write straddles a ring slot: " + r["name"])
        return dest

    def _op_Pad(self, n, ins, att):
        v = self._get(ins[0])
        pads = [int(p) for p in self.consts[ins[1]]]
        assert att.get("mode", b"constant") in (b"constant", "constant")
        assert len(ins) < 3 or not ins[2] or float(self.consts[ins[2]]) == 0.0
        nd = len(v.shape)
        shape = [v.shape[i] + pads[i] + pads[i + nd] for i in range(nd)]
        out = self._new(n.output[0], shape)
        self._ew("fill", out, f0=0.0, label="pad zero")
        inner = View(out.root, sum(pads[i] * out.strides[i] for i in range(nd)), v.shape, out.strides)
        self._copy(v, inner, "pad " + n.output[0])

    # ---------- elementwise ----------
    def _binary(self, n, ins, op):
        a, b = self._get(ins[0]), self._get(ins[1])
        shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape)))
        out = self._new(n.output[0], shape)
        self._ew(op, out, a, b, label=n.op_type + " " + n.output[0])

    def _op_Sub(self, n, ins, att): self._binary(n, ins, "sub")
    def _op_Div(self, n, ins, att): self._binary(n, ins, "div")
    def _op_Max(self, n, ins, att): self._binary(n, ins, "max")

    def _unary(self, n, ins, op, f0=0.0, f1=0.0):
        a = self._get(ins[0])
        out = self._new(n.output[0], a.shape)
        self._ew(op, out, a, f0=f0, f1=f1, label=n.op_type + " " + n.output[0])

    def _op_Log(self, n, ins, att): self._unary(n, ins, "log")

    def _op_Clip(self, n, ins, att):
        lo = float(self.consts[ins[1]]) if len(ins) > 1 and ins[1] else -3.4e38
        hi = float(self.consts[ins[2]]) if len(ins) > 2 and ins[2] else 3.4e38
        self._unary(n, ins, "clip", lo, hi)

    def _op_Pow(self, n, ins, att):
        assert float(self.consts[ins[1]]) == 2.0
        self._unary(n, ins, "sq")

    # ---------- fusion into the producing gemm / conv job ----------
    def _fusable_job(self, name, n_consumers=1):
        """The gemm/conv job whose output `name` is (through views only), if nothing else reads it."""
        if name not in self.chain or len(self.consumers.get(name, [])) != n_consumers:
            return None
        v = self.t[name]
        j = self.producer.get(id(v.root))
        if j is None:
            return None
        if any(v.root is r.root for job in self.jobs[j + 1:] for r in job["reads"] + job["writes"]):
            return None
        return j

    def _fold_bn(self, j, x, sc, sh):
        job = self.jobs[j]
        if job["u"][job["epi_at"]] != EPI["none"] or len(x.shape) < 2:
            return False
        if x.shape[1] != job["N"] or x.strides[1] != job["nstride"]:
            return False
        N = job["N"]
        if job.get("conv") == "dw":
            job["w"][:] = job["w"] * sc[None, :].astype(np.float32)
            job["b"][:N] = job["b"][:N] * sc + sh
        elif job.get("conv"):
            job["w"][:] = (job["w"].reshape(N, -1) * sc[:, None]).ravel()
            job["b"][:N] = job["b"][:N] * sc + sh
        else:
            job["w"][:, :N] *= sc[None, :].astype(np.float32)
            if job["b"] is None:
                bp = np.zeros(job["nb"], np.float32)
                job["u"][job["bias_at"]] = self._w(bp)
                job["b"] = self.last_w
            job["b"][:N] = job["b"][:N] * sc + sh
        return True

    def _set_epi(self, j, epi):
        job = self.jobs[j]
        if job["u"][job["epi_at"]] != EPI["none"]:
            return False
        job["u"][job["epi_at"]] = EPI[epi]
        job["label"] += " +" + epi
        return True

    def _act_pair(self, n, ins, epi, alpha_ok=True):
        """x -> act(x) -> Mul(x, act(x)) (hardswish / silu). Fuse into the producer, or one ew job."""
        x, a = ins[0], n.output[0]
        cs = self.consumers.get(a, [])
        if not (alpha_ok and len(cs) == 1 and cs[0].op_type == "Mul" and set(cs[0].input) == {x, a}):
            return False
        mul = cs[0]
        j = self._fusable_job(x, 2)
        if j is not None and self._set_epi(j, epi):
            self.t[mul.output[0]] = self.t[x]
            self.chain[mul.output[0]] = mul.output[0]
        else:
            v = self._get(x)
            out = self._new(mul.output[0], v.shape)
            self._ew(epi, out, v, f0=1 / 6, f1=0.5, label=epi + " " + mul.output[0])
        self.skip.add(id(mul))
        return True

    def _op_HardSigmoid(self, n, ins, att):
        ok = abs(att.get("alpha", 0.2) - 1 / 6) < 1e-6 and att.get("beta", 0.5) == 0.5
        if not self._act_pair(n, ins, "hswish", ok):
            self._unary(n, ins, "hsig", att.get("alpha", 0.2), att.get("beta", 0.5))

    def _op_Sigmoid(self, n, ins, att):
        if self._act_pair(n, ins, "silu"):
            return
        j = self._fusable_job(ins[0])
        if j is not None and self._set_epi(j, "sigmoid"):
            self.t[n.output[0]] = self.t[ins[0]]
            return
        self._unary(n, ins, "sigmoid")

    def _op_Relu(self, n, ins, att):
        j = self._fusable_job(ins[0])
        if j is not None and self._set_epi(j, "relu"):
            self.t[n.output[0]] = self.t[ins[0]]
            return
        self._unary(n, ins, "relu")

    def _op_Mul(self, n, ins, att):
        """Mul whose only consumer is an Add -> one job out = other + a * b."""
        cs = self.consumers.get(n.output[0], [])
        if len(cs) == 1 and cs[0].op_type == "Add":
            add = cs[0]
            other = add.input[1] if add.input[0] == n.output[0] else add.input[0]
            if other in self.t or other in self.consts:
                a, b, c = self._get(other), self._get(ins[0]), self._get(ins[1])
                shape = list(np.broadcast_shapes(tuple(a.shape), tuple(b.shape), tuple(c.shape)))
                if shape == self.shapes.get(add.output[0], shape):
                    out = self._new(add.output[0], shape)
                    self._ew("addmul", out, a, b, c, label="addmul " + add.output[0])
                    self.skip.add(id(add))
                    return
        self._binary(n, ins, "mul")

    def _op_BatchNormalization(self, n, ins, att):
        x = self._get(ins[0])
        g, b, mean, var = (np.asarray(self.consts[i], np.float64) for i in ins[1:5])
        sc = g / np.sqrt(var + att.get("epsilon", 1e-5))
        sh = b - mean * sc
        j = self._fusable_job(ins[0])
        if j is not None and self._fold_bn(j, x, sc, sh):
            self.t[n.output[0]] = x
            self.chain[n.output[0]] = self.chain[ins[0]]
            return
        cshape = [1, -1] + [1] * (len(x.shape) - 2)
        key = n.output[0] + "#bn"
        self.consts[key + "s"] = sc.astype(np.float32).reshape(cshape)
        self.consts[key + "t"] = sh.astype(np.float32).reshape(cshape)
        out = self._new(n.output[0], x.shape)
        self._ew("fma", out, x, self._wview(key + "s"), self._wview(key + "t"), label="bn " + n.output[0])

    # ---------- matmuls / convs ----------
    def _gemm(self, a, W, bias, out, label, epi="none"):
        """a: view [M, K] (any strides), W: numpy [K, N], bias: numpy [N] or None, out: view [M, N]."""
        M, K = a.shape
        N = W.shape[1]
        NP = (N + 11) // 12 * 12
        if not (a.strides[0] == 1 and a.root.kind == "arena"):   # K-contiguous A: transpose it once (M*K floats)
            st = self._new(label + "#at", [K, M])              # so gemm can load 4 frames as a vec4
            at = View(st.root, 0, [M, K], [1, st.strides[0]])
            self._copy(a, at, "A^T " + label)
            a = at
        wl = np.zeros((K, NP), np.float32)
        wl[:, :N] = W
        woff = self._w(wl)
        wl = self.last_w.reshape(K, NP)                  # the stored copy (BN folding edits it in place)
        boff, bp = NONE, None
        if bias is not None:
            bp = np.zeros(NP, np.float32)
            bp[:N] = bias
            boff = self._w(bp)
            bp = self.last_w
        u = [M, N, K, NP, a, a.strides[0], a.strides[1], woff, boff, out, out.strides[0], out.strides[1], EPI[epi]]
        av4 = int(a.strides[0] == 1 and a.strides[1] % 4 == 0 and a.off % 4 == 0 and a.root.kind == "arena"
                  and a.strides[1] >= (M + 3) // 4 * 4)
        ov4 = int(out.strides[0] == 1 and out.strides[1] % 4 == 0 and out.off % 4 == 0
                  and out.strides[1] >= (M + 3) // 4 * 4)
        self._emit("gemm12_%d%d" % (av4, ov4), u, (NP // 12) * ((M + 3) // 4), [a], [out], label)
        self.jobs[-1].update(w=wl, b=bp, N=N, nstride=out.strides[1], epi_at=12, bias_at=8, nb=NP)
        self.producer[id(out.root)] = len(self.jobs) - 1

    def _flat2(self, v):
        """[.., M, K] view with leading 1s -> 2-D view."""
        sh, st = list(v.shape), list(v.strides)
        while len(sh) > 2:
            assert sh[0] == 1, ("batched", v.shape)
            sh, st = sh[1:], st[1:]
        return View(v.root, v.off, sh, st)

    def _op_Gemm(self, n, ins, att):
        a = self._get(ins[0])
        assert not att.get("transA", 0) and att.get("alpha", 1.0) == 1.0 and att.get("beta", 1.0) == 1.0
        W = self.consts[ins[1]]
        if att.get("transB", 0):
            W = W.T
        bias = self.consts[ins[2]] if len(ins) > 2 and ins[2] else None
        st = self._new(n.output[0] + "#cf", [W.shape[1], a.shape[0]])
        out = View(st.root, 0, [a.shape[0], W.shape[1]], [1, st.strides[0]])
        self.t[n.output[0]] = out
        self._gemm(a, W, bias, out, "gemm " + n.output[0])
        self.chain[n.output[0]] = n.output[0]

    def _op_MatMul(self, n, ins, att):
        a = self._get(ins[0])
        if ins[1] in self.consts:
            W = self.consts[ins[1]]
            assert W.ndim == 2
            a2 = self._flat2(a)
            st = self._new(n.output[0] + "#cf", [W.shape[1], a2.shape[0]])
            out = View(st.root, 0, list(a.shape[:-1]) + [W.shape[1]],
                       [0] * (len(a.shape) - 2) + [1, st.strides[0]])
            self.t[n.output[0]] = out
            self._gemm(a2, W, None, self._flat2(out), "matmul " + n.output[0])
            self.chain[n.output[0]] = n.output[0]
            return
        b = self._get(ins[1])
        self._bmm(a, b, n.output[0], 1.0)

    def _bmm(self, a, b, name, alpha):
        bs = list(np.broadcast_shapes(tuple(a.shape[:-2]), tuple(b.shape[:-2])))
        M, K, N = a.shape[-2], a.shape[-1], b.shape[-1]
        out = self._new(name, bs + [M, N])
        # flatten batch dims (must be expressible with one stride each)
        A = bcast(a, bs + [M, K])
        B = bcast(b, bs + [K, N])
        nb = int(np.prod(bs)) if bs else 1

        def bstride(v):
            dims = [(s, st) for s, st in zip(v.shape[:-2], v.strides[:-2]) if s != 1]
            if not dims:
                return 0
            for (s1, st1), (s2, st2) in zip(dims, dims[1:]):
                assert st1 == st2 * s2, ("batch dims not flat", v.shape, v.strides)
            return dims[-1][1]
        u = [nb, M, N, K, A, bstride(A), A.strides[-2], A.strides[-1], B, bstride(B), B.strides[-2], B.strides[-1],
             out, bstride(out), out.strides[-2], out.strides[-1], f2u(alpha)]
        self._emit("bmm", u, nb * ((M + 3) // 4) * ((N + 3) // 4), [a, b], [out], "bmm " + name)
        return out

    def _op_Einsum(self, n, ins, att):
        eq = att["equation"]
        eq = eq.decode() if isinstance(eq, bytes) else eq
        assert eq == "fm,bft->bmt" and ins[0] in self.consts, eq
        W = self.consts[ins[0]]                          # [F, M]
        x = self._get(ins[1])                            # [1, F, T]
        T = x.shape[2]
        out = self._new(n.output[0], [1, W.shape[1], T])
        a = View(x.root, x.off, [T, x.shape[1]], [x.strides[2], x.strides[1]])        # [T, F]
        o = View(out.root, 0, [T, W.shape[1]], [out.strides[2], out.strides[1]])       # [T, M]
        self._gemm(a, W, None, o, "einsum " + n.output[0])

    def _op_Conv(self, n, ins, att):
        x = self._get(ins[0])                            # [1, Cin, Tin]
        W = self.consts[ins[1]]                          # [Cout, Cin/g, K]
        bias = self.consts[ins[2]] if len(ins) > 2 and ins[2] else None
        assert len(x.shape) == 3 and x.shape[0] == 1 and W.ndim == 3
        G = att.get("group", 1)
        S = att.get("strides", [1])[0]
        D = att.get("dilations", [1])[0]
        pads = att.get("pads", [0, 0])
        Cout, Cg, Kw = W.shape
        Tin = x.shape[2]
        Tout = (Tin + pads[0] + pads[1] - D * (Kw - 1) - 1) // S + 1
        if G == Cout and Cg == 1 and Cout % 4 == 0 and x.strides[1] == 1 and x.strides[2] % 4 == 0 \
                and x.off % 4 == 0 and x.root.kind in ("arena", "fixed"):
            cl = self._dw_consumer(n.output[0])
            out = self._new_cl(n.output[0], [1, Cout, Tout]) if cl else self._new(n.output[0], [1, Cout, Tout])
            self.chain[n.output[0]] = n.output[0]
            wt = np.ascontiguousarray(W[:, 0, :].T)                # [Kw][C]
            woff = self._w(wt)
            Wc = self.last_w.reshape(Kw, Cout)
            boff = self._w(bias if bias is not None else np.zeros(Cout))
            u = [Cout, Tout, Kw, S, D, pads[0], Tin, x, x.strides[2], woff, boff,
                 out, out.strides[1], out.strides[2], EPI["none"], int(cl)]
            self._emit("dwconv1" if S == 1 and D == 1 else "dwconv", u, 0, [x], [out], "dwconv " + n.output[0],
                       groups=groups_for((Cout // 4) * ((Tout + 3) // 4), 256))   # dwconv1: sliding window
            self.jobs[-1].update(w=Wc, b=self.last_w, N=Cout, nstride=out.strides[1], epi_at=14, conv="dw")
            self.producer[id(out.root)] = len(self.jobs) - 1
            return
        out = self._new(n.output[0], [1, Cout, Tout])
        self.chain[n.output[0]] = n.output[0]
        o = View(out.root, 0, [Tout, Cout], [out.strides[2], out.strides[1]])
        if G == 1 and Kw == 1 and S == 1 and pads == [0, 0]:
            a = View(x.root, x.off, [Tin, x.shape[1]], [x.strides[2], x.strides[1]])
            self._gemm(a, W[:, :, 0].T, bias, o, "conv1x1 " + n.output[0])
            return
        if G == 1 and Cg == 1 and D == 1 and pads == [0, 0]:      # framing conv (spectrogram) = gemm on frames
            a = View(x.root, x.off, [Tout, Kw], [S * x.strides[2], x.strides[2]])
            self._gemm(a, W[:, 0, :].T, bias, o, "frames " + n.output[0])
            return
        woff = self._w(W)
        Wc = self.last_w.reshape(Cout, Cg * Kw)          # the stored copy (BN folding edits it in place)
        boff = self._w(bias if bias is not None else np.zeros(Cout))
        bc = self.last_w
        u = [Cout, Tout, Cg, Kw, S, D, pads[0], Tin, x, x.strides[1], x.strides[2], woff, boff,
             out, out.strides[1], out.strides[2], Cout // G, EPI["none"]]
        self._emit("conv1d", u, 0, [x], [out], "conv " + n.output[0], groups=groups_for(Cout * ((Tout + 3) // 4), 256))
        self.jobs[-1].update(w=Wc, b=bc, N=Cout, nstride=out.strides[1], epi_at=17, conv=True)
        self.producer[id(out.root)] = len(self.jobs) - 1

    def _softmax(self, n, ins, att, mode):
        x = self._get(ins[0])
        ax = att.get("axis", -1) % len(x.shape)
        out = self._new(n.output[0], x.shape)
        oth = [i for i in range(len(x.shape)) if i != ax]
        sh = [x.shape[i] for i in oth]
        xs = [x.strides[i] for i in oth]
        os_ = [out.strides[i] for i in oth]
        while len(sh) > 3:
            assert sh[0] == 1
            sh, xs, os_ = sh[1:], xs[1:], os_[1:]
        pad = 3 - len(sh)
        sh, xs, os_ = [1] * pad + sh, [0] * pad + xs, [0] * pad + os_
        u = sh + [x.shape[ax], x] + xs + [x.strides[ax], out] + os_ + [out.strides[ax], mode]
        self._emit("softmax", u, int(np.prod(sh)), [x], [out], n.op_type + " " + n.output[0])   # 256 threads (2026-10-13 sweep)

    def _op_Softmax(self, n, ins, att): self._softmax(n, ins, att, 0)
    def _op_LogSoftmax(self, n, ins, att): self._softmax(n, ins, att, 1)

    def _op_ReduceMax(self, n, ins, att):
        x = self._get(ins[0])
        axes = [int(a) % len(x.shape) for a in self.consts[ins[1]]]
        assert all(x.shape[i] == 1 for i in range(len(x.shape)) if i not in axes), "partial reduce"
        shape = [1 if i in axes else s for i, s in enumerate(x.shape)] if att.get("keepdims", 1) else \
            [s for i, s in enumerate(x.shape) if i not in axes]
        out = self._new(n.output[0], shape)
        sh, st = x.as4()
        self._emit("rmax", sh + [x] + st + [out], 256, [x], [out], "reducemax", groups=(1, 1, 1))

    # ---------- memory ----------
    def _plan_memory(self):
        end = len(self.jobs)
        for j, job in enumerate(self.jobs):
            for v in job["reads"] + job["writes"]:
                r = v.root
                if r.kind != "arena":
                    continue
                r.first = j if r.first is None else min(r.first, j)
                r.last = j if r.last is None else max(r.last, j)
        for v in self.out_views.values():
            v.root.last = end
        arena = [r for r in self.roots if r.kind == "arena" and r.first is not None]
        fixed = [r for r in self.roots if r.kind == "fixed"]
        total = 0
        for r in fixed:                                   # inputs, caches, host uploads: own space
            r.offset = total
            total += (r.size + ALIGN - 1) // ALIGN * ALIGN
        base = total
        placed = []
        for r in sorted(arena, key=lambda r: -r.size):
            size = (r.size + ALIGN - 1) // ALIGN * ALIGN
            if self.debug:
                r.offset = total
                total += size
            else:
                busy = sorted((q.offset, q.offset + (q.size + ALIGN - 1) // ALIGN * ALIGN) for q in placed
                              if not (q.last < r.first or r.last < q.first))
                off = base
                for s, e in busy:
                    if off + size <= s:
                        break
                    off = max(off, e)
                r.offset = off
                total = max(total, off + size)
            placed.append(r)
        self.arena_floats = total

    def addr(self, v, swap=False):
        r = v.root.alias if v.root.kind == "alias" else v.root
        if swap:
            r = self.swap.get(id(r), r)
        return r.offset + v.off

    def plan(self, swap=False):
        n = len(self.jobs)
        pipe = np.array([PIPE[j["pipe"]] for j in self.jobs], np.int32)
        push = np.zeros((n, 32), np.uint32)
        groups = np.array([j["groups"] for j in self.jobs], np.uint32)
        trim = os.environ.get("A4A_TRIM", "1") != "0"
        self.trimmed = 0
        for i, j in enumerate(self.jobs):
            u = j["u"]
            base = j["label"].split(" +")[0].split(" ->")[0]
            if trim and j["pipe"].startswith("gemm12") and base in ROW_TRIM and ROW_TRIM[base] < u[0]:
                m = ROW_TRIM[base]
                u = [m] + list(u[1:])
                groups[i] = groups_for((u[3] // 12) * ((m + 3) // 4))
                self.trimmed += 1
            vals = []
            for x in u:
                if isinstance(x, View):
                    vals.append(self.addr(x, swap))
                else:
                    vals.append(int(x) & 0xFFFFFFFF)
            push[i, :len(vals)] = np.array(vals, np.int64).astype(np.uint32)
        weights = np.concatenate(self.wparts).astype(np.float32)
        return dict(pipe=pipe, push=push, groups=groups, weights=weights, arena_floats=self.arena_floats,
                    labels=[j["label"] for j in self.jobs],
                    init=[(self.addr(v), np.asarray(arr, np.float32).ravel()) for v, arr in self.init_data])


class A4AGPU:
    """One stream graph on the GPU. step(pcm, filter_cache=None, attn_cache=None) -> (log_probs, hidden).
    Caches stay on the GPU; pass numpy caches only to (re)load state (reset / restore)."""
    def __init__(self, path=os.path.join(ONNX_DIR, "stream_c16r4.onnx"), debug=False):
        for lib in ("libwayland-client.so.0", "libxshmfence.so.1", "libvulkan.so.1") if VK else ():
            try:
                ctypes.CDLL(os.path.join(VK, "lib", lib), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        icd = os.path.join(VK, "icd/broadcom_icd.json")
        if os.path.exists(icd):                          # else the system's Vulkan driver list (Raspberry Pi OS)
            os.environ.setdefault("VK_ICD_FILENAMES", icd)
        t = time.time()
        self.c = Compiler(path, debug)
        self.p = self.c.plan()
        self.build_s = time.time() - t
        self.lib = ctypes.CDLL(os.path.join(HERE, "libygpu.so"))
        self.lib.yg_arena.restype = ctypes.c_void_p
        names = (ctypes.c_char_p * len(SHADERS))(*[s.encode() for s in SHADERS])
        w = self.p["weights"]
        # real work is never inside an assert: python -O would skip it (Codex audit 2026-10-14)
        ret = self.lib.yg_init(os.path.join(HERE, "shaders").encode(), names, len(SHADERS),
                               ctypes.c_long(self.p["arena_floats"]), w.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_long(w.size))
        if ret == -100:
            raise RuntimeError("libygpu.so already in use: only one A4AGPU per process")
        if ret != 0:
            raise RuntimeError(f"yg_init failed ({ret})")
        self.arena = np.ctypeslib.as_array(ctypes.cast(self.lib.yg_arena(), ctypes.POINTER(ctypes.c_float)),
                                           shape=(self.p["arena_floats"],))
        for a, arr in self.p["init"]:                    # small constants live in the arena
            self.arena[a:a + arr.size] = arr
        self.par = 0                                     # which plan runs next (rings: A/B swap every step)
        plans = [self.p] + ([self.c.plan(swap=True)] if self.c.rings else [])
        for s, pl in enumerate(plans):
            pipe, push, groups = pl["pipe"], pl["push"], pl["groups"]
            if self.c.rings and self.lib.yg_slot(s) != 0:
                raise RuntimeError("libygpu.so has no second plan slot (rebuild it)")
            if self.lib.yg_plan(len(pipe), pipe.ctypes.data_as(ctypes.c_void_p), push.ctypes.data_as(ctypes.c_void_p),
                                groups.ctypes.data_as(ctypes.c_void_p)) != 0:
                raise RuntimeError("yg_plan failed")
        if self.c.rings:
            self.lib.yg_slot(0)
        self.gpu_ms = 0.0

    def region(self, name):
        v = self.c.in_views[name]
        a = self.c.addr(v)
        return self.arena[a:a + v.size()]

    def read(self, v):
        a = self.c.addr(v)
        return np.lib.stride_tricks.as_strided(self.arena[a:], shape=v.shape,
                                               strides=[s * 4 for s in v.strides]).copy()

    def _host(self):
        c = self.c
        attn = self.region("attn_cache")
        vals = {}
        for n in c.host_nodes:
            att = c.attrs[id(n)]
            if n in c.attn_seeds:
                ins = [None] + [c.consts.get(x) for x in n.input[1:]]
                ins[0] = attn                               # numpy view (reads a few floats)
                outs = [np_op(n, ins, att)]
            else:
                ins = [c.consts.get(x, vals.get(x)) if x else None for x in n.input]
                outs = c._np_outputs(n, ins, att)
            for o, v in zip(n.output, outs):
                vals[o] = np.array(v)
        for name, view in c.host_up.items():
            a = c.addr(view)
            self.arena[a:a + view.size()] = np.asarray(vals[name], np.float32).ravel()

    def _ring_rows(self, r):
        """This step's history of ring r (A on even plans, B on odd), as a flat [T][C] arena slice."""
        root = r["A"] if self.par == 0 else r["B"]
        return self.arena[root.offset:root.offset + r["n"]]

    def load_cache(self, which, arr):
        x = self.c.to_internal(which, arr)
        self.region(which)[:] = x
        if which == "filter_cache":
            for r in self.c.rings:
                self._ring_rows(r)[:] = x[r["off"]:r["off"] + r["n"]]

    def cache(self, which):
        """A cache in ONNX layout (the GPU keeps filter_cache slots transposed, ring slots in their A/B buffer)."""
        x = np.array(self.region(which))
        if which == "filter_cache":
            for r in self.c.rings:
                x[r["off"]:r["off"] + r["n"]] = self._ring_rows(r)
        return self.c.from_internal(which, x)

    def step(self, pcm, filter_cache=None, attn_cache=None):
        if filter_cache is not None:
            self.load_cache("filter_cache", filter_cache)
        if attn_cache is not None:
            self.load_cache("attn_cache", attn_cache)
        self.region(self.c.pcm_name)[:] = np.asarray(pcm, np.float32).ravel()
        self._host()
        t = ctypes.c_double()
        if self.c.rings:
            self.lib.yg_slot(self.par)
        if self.lib.yg_run(ctypes.byref(t)) != 0:
            raise RuntimeError("yg_run failed")
        if self.c.rings:
            self.par ^= 1
        self.gpu_ms = t.value
        return {k: self.read(v) for k, v in self.c.out_views.items()}

    def caches(self):
        return self.cache("filter_cache"), self.cache("attn_cache")

    def job_times(self):
        ms = (ctypes.c_double * len(self.p["pipe"]))()
        self.lib.yg_times(ms)
        return list(ms)


class GpuCache:
    """Stands in for a cache that lives on the GPU. deepcopy (the dictation script's early-finish snapshot) reads
    it back in ONNX layout; a numpy array handed to run_stream later is loaded back onto the GPU."""
    def __init__(self, g, which):
        self.g, self.which = g, which

    def __deepcopy__(self, memo):
        return self.g.cache(self.which)


def use_gpu(engine, tier="stream_c16r4"):
    """Run `tier` of a FUTO asr4all_onnx Engine (asr4all-m) on the GPU; other tiers and PCEC stay on the CPU."""
    g = A4AGPU()
    names = list(g.c.out_views)
    lp = next(n for n in names if n.endswith("log_probs"))
    hid = next(n for n in names if n.endswith("hidden"))
    orig = engine.model.run_stream

    def run_stream(t, win, fc, ac):
        if t["name"] != tier:
            return orig(t, win, fc, ac)
        outs = g.step(win, None if isinstance(fc, GpuCache) else fc, None if isinstance(ac, GpuCache) else ac)
        return [outs[lp], GpuCache(g, "filter_cache"), GpuCache(g, "attn_cache"), outs[hid]]
    engine.model.run_stream = run_stream
    return g


def _check(steps=6):
    import onnxruntime as ort
    path = os.path.join(ONNX_DIR, "stream_c16r4.onnx")
    debug = "--debug" in sys.argv
    g = A4AGPU(path, debug)
    c = g.c
    print(f"compiled in {g.build_s:.1f} s: {len(c.jobs)} jobs, {len(c.host_nodes)} host nodes "
          f"({len(c.host_up)} uploads), arena {c.arena_floats * 4 / 1e6:.0f} MB, weights {g.p['weights'].nbytes / 1e6:.0f} MB")
    from collections import Counter
    print("jobs:", dict(Counter(j["pipe"] for j in c.jobs)), "gemms into concat:", c.redirected)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 3
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    import wave
    clip = os.path.expanduser(sys.argv[sys.argv.index("--wav") + 1]) if "--wav" in sys.argv else None
    if clip:
        with wave.open(clip) as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    else:
        pcm = np.random.default_rng(0).standard_normal(16000 * 12).astype(np.float32) * 0.05
    fc = np.zeros(1728000, np.float32)
    ac = np.zeros(548354, np.float32)
    first = True
    win = 24960
    names = [o.name for o in s.get_outputs()]
    for k in range(steps):
        start = k * 16 * 640
        w = np.zeros(win, np.float32)
        seg = pcm[start:start + win]
        w[:len(seg)] = seg
        ref = s.run(None, {c.pcm_name: w[None], "filter_cache": fc, "attn_cache": ac})
        t = time.perf_counter()
        got = g.step(w, fc if first else None, ac if first else None)
        wall = (time.perf_counter() - t) * 1e3
        first = False
        fc, ac = ref[1], ref[2]
        gfc, gac = g.caches()
        lp, lpg = ref[0], got[names[0]]
        msg = [f"step {k}: gpu {g.gpu_ms:.0f} ms wall {wall:.0f} ms"]
        msg.append(f"logp max diff {np.abs(lp - lpg).max():.1e}")
        msg.append(f"argmax same {(lp.argmax(-1) == lpg.argmax(-1)).mean() * 100:.0f}%")
        msg.append(f"hidden {np.abs(ref[3] - got[names[3]]).max():.1e}")
        msg.append(f"fcache {np.abs(fc - gfc).max():.1e} acache {np.abs(ac - gac).max():.1e}")
        print("  ".join(msg), flush=True)
        if "--sync" in sys.argv:                 # keep GPU on the reference caches (isolates one step's error)
            g.load_cache("filter_cache", fc)
            g.load_cache("attn_cache", ac)
    if "--profile" in sys.argv:
        ms = g.job_times()
        by = Counter()
        for j, m in zip(c.jobs, ms):
            by[j["pipe"]] += m
        print("GPU ms by shader:", {k: round(v, 1) for k, v in by.most_common()}, "total", round(sum(ms)))
        top = sorted(zip(ms, [j["label"] for j in c.jobs]), reverse=True)[:12]
        for m, l in top:
            print(f"  {m:6.2f} ms  {l}")


if __name__ == "__main__":
    _check(int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 6)
