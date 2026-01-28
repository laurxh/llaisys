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

from ctypes import byref, c_int, c_int64, c_size_t, c_void_p

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
)


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

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
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

        meta = LlaisysQwen2Meta(
            dtype=dtype,
            nlayer=nlayer,
            hs=hs,
            nh=nh,
            nkvh=nkvh,
            dh=dh,
            di=di,
            maxseq=maxseq,
            voc=voc,
            epsilon=epsilon,
            theta=theta,
            end_token=end_token,
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(byref(meta), c_int(device), device_ids, 1)
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        self._meta = meta
        self._device = device
        self._end_token = end_token

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_ptr:
            raise RuntimeError("Failed to get model weights")
        self._weights_ptr = weights_ptr
        self._weights: LlaisysQwen2Weights = weights_ptr.contents

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
                        _load_tensor(self._weights.in_embed, in_embed_array)
                        loaded["in_embed"] = True
                        continue
                    if name_ in ("lm_head.weight", "model.lm_head.weight"):
                        _load_tensor_allow_transpose(self._weights.out_embed, data_.get_tensor(name_))
                        loaded["out_embed"] = True
                        continue
                    if name_ in ("model.norm.weight", "norm.weight"):
                        _load_tensor(self._weights.out_norm_w, data_.get_tensor(name_))
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
                        _load_tensor(self._weights.attn_norm_w[layer], data_.get_tensor(name_))
                        loaded["attn_norm_w"][layer] = True
                    elif suffix == "self_attn.q_proj.weight":
                        _load_tensor(self._weights.attn_q_w[layer], data_.get_tensor(name_))
                        loaded["attn_q_w"][layer] = True
                    elif suffix == "self_attn.q_proj.bias":
                        _load_tensor(self._weights.attn_q_b[layer], data_.get_tensor(name_))
                        bias_loaded["attn_q_b"][layer] = True
                    elif suffix == "self_attn.k_proj.weight":
                        _load_tensor(self._weights.attn_k_w[layer], data_.get_tensor(name_))
                        loaded["attn_k_w"][layer] = True
                    elif suffix == "self_attn.k_proj.bias":
                        _load_tensor(self._weights.attn_k_b[layer], data_.get_tensor(name_))
                        bias_loaded["attn_k_b"][layer] = True
                    elif suffix == "self_attn.v_proj.weight":
                        _load_tensor(self._weights.attn_v_w[layer], data_.get_tensor(name_))
                        loaded["attn_v_w"][layer] = True
                    elif suffix == "self_attn.v_proj.bias":
                        _load_tensor(self._weights.attn_v_b[layer], data_.get_tensor(name_))
                        bias_loaded["attn_v_b"][layer] = True
                    elif suffix == "self_attn.o_proj.weight":
                        _load_tensor(self._weights.attn_o_w[layer], data_.get_tensor(name_))
                        loaded["attn_o_w"][layer] = True
                    elif suffix == "post_attention_layernorm.weight":
                        _load_tensor(self._weights.mlp_norm_w[layer], data_.get_tensor(name_))
                        loaded["mlp_norm_w"][layer] = True
                    elif suffix == "mlp.gate_proj.weight":
                        _load_tensor(self._weights.mlp_gate_w[layer], data_.get_tensor(name_))
                        loaded["mlp_gate_w"][layer] = True
                    elif suffix == "mlp.up_proj.weight":
                        _load_tensor(self._weights.mlp_up_w[layer], data_.get_tensor(name_))
                        loaded["mlp_up_w"][layer] = True
                    elif suffix == "mlp.down_proj.weight":
                        _load_tensor(self._weights.mlp_down_w[layer], data_.get_tensor(name_))
                        loaded["mlp_down_w"][layer] = True

        missing = []
        if not loaded["in_embed"]:
            missing.append("in_embed")
        if not loaded["out_norm_w"]:
            missing.append("out_norm_w")

        if not loaded["out_embed"] and loaded["in_embed"]:
            if in_embed_array is None:
                raise RuntimeError("in_embed loaded but source array missing for tying out_embed")
            _load_tensor(self._weights.out_embed, in_embed_array)
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

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _infer(self, tokens):
        arr = (c_int64 * len(tokens))(*tokens)
        return int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, c_size_t(len(tokens))))

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
        if len(inputs) == 0:
            return []

        tokens = list(inputs)
        next_token = self._infer(tokens)
        tokens.append(next_token)

        for _ in range(max_new_tokens - 1):
            next_token = self._infer([next_token])
            tokens.append(next_token)
            if next_token == self._end_token:
                break

        return tokens
