from typing import Sequence
from pathlib import Path
import json
import re

import numpy as np
import safetensors

try:
    import torch
except Exception:  # pragma: no cover - optional fallback
    torch = None

from ctypes import POINTER, c_int64, c_size_t, c_void_p, cast, memmove

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
)
from ..tensor import Tensor


def _dtype_from_numpy(np_dtype):
    if np_dtype == np.float16:
        return DataType.F16
    if np_dtype == np.float32:
        return DataType.F32
    if np_dtype == np.int64:
        return DataType.I64
    if str(np_dtype) == "bfloat16":
        return DataType.BF16
    raise ValueError(f"Unsupported numpy dtype: {np_dtype}")


def _dtype_from_config(config):
    dtype = config.get("torch_dtype")
    if dtype is None:
        return None
    if isinstance(dtype, str):
        if dtype in ("bfloat16", "bf16"):
            return DataType.BF16
        if dtype in ("float16", "fp16", "f16"):
            return DataType.F16
        if dtype in ("float32", "fp32", "f32"):
            return DataType.F32
    return None


def _tensor_shape(tensor_handle):
    ndim = int(LIB_LLAISYS.tensorGetNdim(tensor_handle))
    buf = (c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(tensor_handle, buf)
    return tuple(buf[i] for i in range(ndim))


def _tensor_dtype(tensor_handle):
    return DataType(LIB_LLAISYS.tensorGetDataType(tensor_handle))


def _to_load_array(tensor_handle, array):
    target_dtype = _tensor_dtype(tensor_handle)

    if torch is not None and isinstance(array, torch.Tensor):
        if target_dtype == DataType.BF16:
            array = array.to(torch.bfloat16)
            return array.contiguous().view(torch.uint16).cpu().numpy()
        if target_dtype == DataType.F16:
            return array.to(torch.float16).contiguous().cpu().numpy()
        if target_dtype == DataType.F32:
            return array.to(torch.float32).contiguous().cpu().numpy()
        if target_dtype == DataType.I64:
            return array.to(torch.int64).contiguous().cpu().numpy()
        return array.contiguous().cpu().numpy()

    arr = np.ascontiguousarray(array)
    if target_dtype == DataType.BF16:
        if str(arr.dtype) == "bfloat16":
            return arr.view(np.uint16)
        if torch is None:
            raise TypeError("bfloat16 weights require torch for conversion")
        tensor = torch.from_numpy(arr).to(torch.bfloat16)
        return tensor.contiguous().view(torch.uint16).cpu().numpy()
    if target_dtype == DataType.F16:
        return arr.astype(np.float16, copy=False)
    if target_dtype == DataType.F32:
        return arr.astype(np.float32, copy=False)
    if target_dtype == DataType.I64:
        return arr.astype(np.int64, copy=False)
    return arr


def _load_tensor(tensor_handle, array):
    arr = _to_load_array(tensor_handle, array)
    expected = _tensor_shape(tensor_handle)
    if expected != arr.shape:
        raise ValueError(f"Shape mismatch for tensor load: expected {expected}, got {arr.shape}")
    LIB_LLAISYS.tensorLoad(tensor_handle, c_void_p(arr.ctypes.data))


def _load_tensor_allow_transpose(tensor_handle, array):
    try:
        _load_tensor(tensor_handle, array)
        return
    except ValueError:
        arr = np.ascontiguousarray(array)
        expected = _tensor_shape(tensor_handle)
        if arr.ndim == 2 and arr.T.shape == expected:
            _load_tensor(tensor_handle, arr.T)
            return
        raise


class Qwen2:
    """Minimal single-sequence Qwen2 runner for test/test_infer.py."""

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        if device != DeviceType.CPU:
            raise NotImplementedError("Minimal Qwen2 Python runner only supports CPU")
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        nlayer = int(config.get("num_hidden_layers", config.get("n_layer", 0)))
        hs = int(config.get("hidden_size", 0))
        nh = int(config.get("num_attention_heads", 0))
        nkvh = int(config.get("num_key_value_heads", nh))
        di = int(config.get("intermediate_size", 0))
        maxseq = int(config.get("max_position_embeddings", config.get("max_seq_len", 0)))
        voc = int(config.get("vocab_size", 0))
        epsilon = float(config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-5)))
        theta = float(config.get("rope_theta", config.get("rotary_emb_base", 10000.0)))
        end_token = config.get("eos_token_id", -1)
        if isinstance(end_token, list):
            end_token = end_token[0] if end_token else -1
        end_token = int(end_token)

        dtype = _dtype_from_config(config)
        if dtype is None:
            for file in sorted(model_path.glob("*.safetensors")):
                with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                    for name_ in data_.keys():
                        candidate = _dtype_from_numpy(data_.get_tensor(name_).dtype)
                        if candidate != DataType.I64:
                            dtype = candidate
                            break
                if dtype is not None:
                    break
        if dtype is None:
            raise ValueError("Failed to infer model dtype")

        if hs == 0 or nh == 0 or nlayer == 0 or di == 0 or maxseq == 0 or voc == 0:
            raise ValueError("Invalid config: missing model dimensions")

        dh = int(config.get("head_dim", hs // nh))

        self._meta = type("Qwen2Meta", (), {})()
        self._meta.dtype = dtype
        self._meta.nlayer = nlayer
        self._meta.hs = hs
        self._meta.nh = nh
        self._meta.nkvh = nkvh
        self._meta.dh = dh
        self._meta.di = di
        self._meta.maxseq = maxseq
        self._meta.voc = voc
        self._meta.epsilon = epsilon
        self._meta.theta = theta
        self._meta.end_token = end_token
        self._device = device
        self._dtype = DataType(self._meta.dtype)
        self._end_token = end_token
        self._scale = float(1.0 / np.sqrt(float(self._meta.dh)))
        self._cur_len = 0

        self._k_cache = []
        self._v_cache = []
        for _ in range(nlayer):
            self._k_cache.append(Tensor((self._meta.maxseq, self._meta.nkvh, self._meta.dh), self._dtype, self._device, 0))
            self._v_cache.append(Tensor((self._meta.maxseq, self._meta.nkvh, self._meta.dh), self._dtype, self._device, 0))

        self._in_embed = Tensor((voc, hs), self._dtype, self._device, 0)
        self._out_embed = Tensor((voc, hs), self._dtype, self._device, 0)
        self._out_norm_w = Tensor((hs,), self._dtype, self._device, 0)
        self._attn_norm_w = [Tensor((hs,), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_q_w = [Tensor((nh * dh, hs), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_q_b = [Tensor((nh * dh,), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_k_w = [Tensor((nkvh * dh, hs), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_k_b = [Tensor((nkvh * dh,), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_v_w = [Tensor((nkvh * dh, hs), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_v_b = [Tensor((nkvh * dh,), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._attn_o_w = [Tensor((hs, nh * dh), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._mlp_norm_w = [Tensor((hs,), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._mlp_gate_w = [Tensor((di, hs), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._mlp_up_w = [Tensor((di, hs), self._dtype, self._device, 0) for _ in range(nlayer)]
        self._mlp_down_w = [Tensor((hs, di), self._dtype, self._device, 0) for _ in range(nlayer)]

        loaded = {
            "in_embed": False,
            "out_embed": False,
            "out_norm_w": False,
            "attn_norm_w": [False] * nlayer,
            "attn_q_w": [False] * nlayer,
            "attn_k_w": [False] * nlayer,
            "attn_v_w": [False] * nlayer,
            "attn_o_w": [False] * nlayer,
            "mlp_norm_w": [False] * nlayer,
            "mlp_gate_w": [False] * nlayer,
            "mlp_up_w": [False] * nlayer,
            "mlp_down_w": [False] * nlayer,
        }
        bias_loaded = {
            "attn_q_b": [False] * nlayer,
            "attn_k_b": [False] * nlayer,
            "attn_v_b": [False] * nlayer,
        }
        in_embed_array = None

        layer_re = re.compile(r"(?:model\.)?layers\.(\d+)\.(.+)")

        framework = "pt" if dtype == DataType.BF16 else "numpy"
        for file in sorted(model_path.glob("*.safetensors")):
            if framework == "pt" and torch is None:
                raise RuntimeError("torch is required to load bfloat16 safetensors")
            with safetensors.safe_open(file, framework=framework, device="cpu") as data_:
                for name_ in data_.keys():
                    if name_ in ("model.embed_tokens.weight", "embed_tokens.weight"):
                        in_embed_array = data_.get_tensor(name_)
                        _load_tensor(self._in_embed.lib_tensor(), in_embed_array)
                        loaded["in_embed"] = True
                        continue
                    if name_ in ("lm_head.weight", "model.lm_head.weight"):
                        _load_tensor_allow_transpose(self._out_embed.lib_tensor(), data_.get_tensor(name_))
                        loaded["out_embed"] = True
                        continue
                    if name_ in ("model.norm.weight", "norm.weight"):
                        _load_tensor(self._out_norm_w.lib_tensor(), data_.get_tensor(name_))
                        loaded["out_norm_w"] = True
                        continue

                    match = layer_re.match(name_)
                    if not match:
                        continue

                    layer = int(match.group(1))
                    suffix = match.group(2)

                    if layer < 0 or layer >= nlayer:
                        continue

                    if suffix == "input_layernorm.weight":
                        _load_tensor(self._attn_norm_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["attn_norm_w"][layer] = True
                    elif suffix == "self_attn.q_proj.weight":
                        _load_tensor(self._attn_q_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["attn_q_w"][layer] = True
                    elif suffix == "self_attn.q_proj.bias":
                        _load_tensor(self._attn_q_b[layer].lib_tensor(), data_.get_tensor(name_))
                        bias_loaded["attn_q_b"][layer] = True
                    elif suffix == "self_attn.k_proj.weight":
                        _load_tensor(self._attn_k_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["attn_k_w"][layer] = True
                    elif suffix == "self_attn.k_proj.bias":
                        _load_tensor(self._attn_k_b[layer].lib_tensor(), data_.get_tensor(name_))
                        bias_loaded["attn_k_b"][layer] = True
                    elif suffix == "self_attn.v_proj.weight":
                        _load_tensor(self._attn_v_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["attn_v_w"][layer] = True
                    elif suffix == "self_attn.v_proj.bias":
                        _load_tensor(self._attn_v_b[layer].lib_tensor(), data_.get_tensor(name_))
                        bias_loaded["attn_v_b"][layer] = True
                    elif suffix == "self_attn.o_proj.weight":
                        _load_tensor(self._attn_o_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["attn_o_w"][layer] = True
                    elif suffix == "post_attention_layernorm.weight":
                        _load_tensor(self._mlp_norm_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["mlp_norm_w"][layer] = True
                    elif suffix == "mlp.gate_proj.weight":
                        _load_tensor(self._mlp_gate_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["mlp_gate_w"][layer] = True
                    elif suffix == "mlp.up_proj.weight":
                        _load_tensor(self._mlp_up_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["mlp_up_w"][layer] = True
                    elif suffix == "mlp.down_proj.weight":
                        _load_tensor(self._mlp_down_w[layer].lib_tensor(), data_.get_tensor(name_))
                        loaded["mlp_down_w"][layer] = True

        missing = []
        if not loaded["in_embed"]:
            missing.append("in_embed")
        if not loaded["out_norm_w"]:
            missing.append("out_norm_w")

        if not loaded["out_embed"] and loaded["in_embed"]:
            if in_embed_array is None:
                raise RuntimeError("in_embed loaded but source array missing for tying out_embed")
            _load_tensor(self._out_embed.lib_tensor(), in_embed_array)
            loaded["out_embed"] = True

        if not loaded["out_embed"]:
            missing.append("out_embed")

        for i in range(nlayer):
            for key in (
                "attn_norm_w",
                "attn_q_w",
                "attn_k_w",
                "attn_v_w",
                "attn_o_w",
                "mlp_norm_w",
                "mlp_gate_w",
                "mlp_up_w",
                "mlp_down_w",
            ):
                if not loaded[key][i]:
                    missing.append(f"{key}[{i}]")
        if missing:
            raise RuntimeError(f"Missing model weights: {', '.join(missing[:5])} ...")

        # q/k/v bias might be absent in some checkpoints; default to zero.
        zero_q = np.zeros((nh * dh,), dtype=np.float32)
        zero_kv = np.zeros((nkvh * dh,), dtype=np.float32)
        for i in range(nlayer):
            if not bias_loaded["attn_q_b"][i]:
                _load_tensor_allow_transpose(self._attn_q_b[i].lib_tensor(), zero_q)
            if not bias_loaded["attn_k_b"][i]:
                _load_tensor_allow_transpose(self._attn_k_b[i].lib_tensor(), zero_kv)
            if not bias_loaded["attn_v_b"][i]:
                _load_tensor_allow_transpose(self._attn_v_b[i].lib_tensor(), zero_kv)

    def __del__(self):
        self._k_cache = []
        self._v_cache = []

    @staticmethod
    def _dtype_nbytes(dtype: DataType) -> int:
        if dtype in (DataType.BF16, DataType.F16, DataType.I16, DataType.U16):
            return 2
        if dtype in (DataType.F32, DataType.I32, DataType.U32):
            return 4
        if dtype in (DataType.I64, DataType.U64, DataType.F64):
            return 8
        if dtype in (DataType.BYTE, DataType.BOOL, DataType.I8, DataType.U8):
            return 1
        raise ValueError(f"Unsupported dtype element size: {dtype}")

    def _load_i64_scalar(self, tensor: Tensor, value: int):
        arr = np.asarray([value], dtype=np.int64)
        tensor.load(c_void_p(arr.ctypes.data))

    def _new(self, *shape: int, dtype: DataType | None = None) -> Tensor:
        return Tensor(shape, self._dtype if dtype is None else dtype, self._device, 0)

    def _write_cache(self, cache: Tensor, src: Tensor, pos: int):
        per_token = int(self._meta.nkvh * self._meta.dh)
        elem_size = self._dtype_nbytes(self._dtype)
        copy_bytes = per_token * elem_size
        dst = int(cast(cache.data_ptr(), c_void_p).value) + pos * copy_bytes
        src_ptr = cast(src.data_ptr(), c_void_p)
        memmove(dst, src_ptr, copy_bytes)

    def _forward_layer(self, x: Tensor, pos_ids: Tensor, layer: int) -> Tensor:
        x_norm = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysRmsNorm(
            x_norm.lib_tensor(), x.lib_tensor(), self._attn_norm_w[layer].lib_tensor(), float(self._meta.epsilon)
        )

        q_lin = self._new(1, self._meta.nh * self._meta.dh)
        k_lin = self._new(1, self._meta.nkvh * self._meta.dh)
        v_lin = self._new(1, self._meta.nkvh * self._meta.dh)
        LIB_LLAISYS.llaisysLinear(
            q_lin.lib_tensor(), x_norm.lib_tensor(), self._attn_q_w[layer].lib_tensor(), self._attn_q_b[layer].lib_tensor()
        )
        LIB_LLAISYS.llaisysLinear(
            k_lin.lib_tensor(), x_norm.lib_tensor(), self._attn_k_w[layer].lib_tensor(), self._attn_k_b[layer].lib_tensor()
        )
        LIB_LLAISYS.llaisysLinear(
            v_lin.lib_tensor(), x_norm.lib_tensor(), self._attn_v_w[layer].lib_tensor(), self._attn_v_b[layer].lib_tensor()
        )

        q_rope = self._new(1, self._meta.nh, self._meta.dh)
        k_rope = self._new(1, self._meta.nkvh, self._meta.dh)
        q_view = q_lin.view(1, self._meta.nh, self._meta.dh)
        k_view = k_lin.view(1, self._meta.nkvh, self._meta.dh)
        LIB_LLAISYS.llaisysROPE(
            q_rope.lib_tensor(), q_view.lib_tensor(), pos_ids.lib_tensor(), float(self._meta.theta)
        )
        v = v_lin.view(1, self._meta.nkvh, self._meta.dh)
        LIB_LLAISYS.llaisysROPE(
            k_rope.lib_tensor(), k_view.lib_tensor(), pos_ids.lib_tensor(), float(self._meta.theta)
        )

        self._write_cache(self._k_cache[layer], k_rope, self._cur_len)
        self._write_cache(self._v_cache[layer], v, self._cur_len)
        k_slice = self._k_cache[layer].slice(0, 0, self._cur_len + 1)
        v_slice = self._v_cache[layer].slice(0, 0, self._cur_len + 1)
        k_cache = k_slice.contiguous()
        v_cache = v_slice.contiguous()

        attn = self._new(1, self._meta.nh, self._meta.dh)
        LIB_LLAISYS.llaisysSelfAttention(
            attn.lib_tensor(), q_rope.lib_tensor(), k_cache.lib_tensor(), v_cache.lib_tensor(), self._scale
        )
        attn_proj = self._new(1, self._meta.hs)
        attn_2d = attn.view(1, self._meta.hs)
        LIB_LLAISYS.llaisysLinear(
            attn_proj.lib_tensor(), attn_2d.lib_tensor(), self._attn_o_w[layer].lib_tensor(), None
        )

        x_attn = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysAdd(x_attn.lib_tensor(), attn_proj.lib_tensor(), x.lib_tensor())

        mlp_norm = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysRmsNorm(
            mlp_norm.lib_tensor(), x_attn.lib_tensor(), self._mlp_norm_w[layer].lib_tensor(), float(self._meta.epsilon)
        )
        gate = self._new(1, self._meta.di)
        up = self._new(1, self._meta.di)
        swiglu_out = self._new(1, self._meta.di)
        down = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysLinear(gate.lib_tensor(), mlp_norm.lib_tensor(), self._mlp_gate_w[layer].lib_tensor(), None)
        LIB_LLAISYS.llaisysLinear(up.lib_tensor(), mlp_norm.lib_tensor(), self._mlp_up_w[layer].lib_tensor(), None)
        LIB_LLAISYS.llaisysSwiGLU(swiglu_out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())
        LIB_LLAISYS.llaisysLinear(down.lib_tensor(), swiglu_out.lib_tensor(), self._mlp_down_w[layer].lib_tensor(), None)

        x_next = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysAdd(x_next.lib_tensor(), down.lib_tensor(), x_attn.lib_tensor())
        return x_next

    def _forward_one(self, token: int) -> int:
        idx = self._new(1, dtype=DataType.I64)
        pos_ids = self._new(1, dtype=DataType.I64)
        self._load_i64_scalar(idx, token)
        self._load_i64_scalar(pos_ids, self._cur_len)

        x = self._new(1, self._meta.hs)
        LIB_LLAISYS.llaisysEmbedding(x.lib_tensor(), idx.lib_tensor(), self._in_embed.lib_tensor())
        for layer in range(int(self._meta.nlayer)):
            x = self._forward_layer(x, pos_ids, layer)

        final_norm = self._new(1, self._meta.hs)
        logits = self._new(1, self._meta.voc)
        LIB_LLAISYS.llaisysRmsNorm(
            final_norm.lib_tensor(), x.lib_tensor(), self._out_norm_w.lib_tensor(), float(self._meta.epsilon)
        )
        LIB_LLAISYS.llaisysLinear(logits.lib_tensor(), final_norm.lib_tensor(), self._out_embed.lib_tensor(), None)

        max_idx = self._new(1, dtype=DataType.I64)
        max_val = self._new(1)
        LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), logits.lib_tensor())
        self._cur_len += 1
        return int(cast(max_idx.data_ptr(), POINTER(c_int64))[0])

    def _prefill(self, tokens: Sequence[int]) -> int:
        if len(tokens) == 0:
            raise ValueError("tokens must not be empty")
        if len(tokens) > int(self._meta.maxseq):
            raise ValueError("sequence length exceeds maxseq")
        next_token = -1
        for token in tokens:
            next_token = self._forward_one(int(token))
        return next_token

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens <= 0:
            return list(inputs)
        if len(inputs) == 0:
            return []
        # Keep signature compatible with test_infer.py; decoding is greedy only.
        _ = (top_k, top_p, temperature)

        self._cur_len = 0
        tokens = list(inputs)
        next_token = self._prefill(tokens)
        tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            if self._cur_len >= int(self._meta.maxseq):
                break
            next_token = self._forward_one(next_token)
            tokens.append(next_token)
            if next_token == self._end_token:
                break

        return tokens
