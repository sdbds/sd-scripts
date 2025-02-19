import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import einops
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Gemma2Config, Gemma2Model

from library.utils import setup_logging
from library import lumina_models, flux_models
from library.utils import load_safetensors
import logging

setup_logging()
logger = logging.getLogger(__name__)

MODEL_VERSION_LUMINA_V2 = "lumina2"

def load_lumina_model(
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
    disable_mmap: bool = False,
):
    """
    Load the Lumina model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (torch.device): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.

    Returns:
        model (lumina_models.NextDiT): The loaded model.
    """
    logger.info("Building Lumina")
    with torch.device("meta"):
        model = lumina_models.NextDiT_2B_GQA_patch2_Adaln_Refiner().to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    state_dict = load_safetensors(
        ckpt_path, device=device, disable_mmap=disable_mmap, dtype=dtype
    )
    info = model.load_state_dict(state_dict, strict=False, assign=True)
    logger.info(f"Loaded Lumina: {info}")
    return model

def load_ae(
    ckpt_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> flux_models.AutoEncoder:
    """
    Load the AutoEncoder model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (Union[str, torch.device]): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.

    Returns:
        ae (flux_models.AutoEncoder): The loaded model.
    """
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        # dev and schnell have the same AE params
        ae = flux_models.AutoEncoder(flux_models.configs["schnell"].ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(
        ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype
    )
    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


def load_gemma2(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> Gemma2Model:
    """
    Load the Gemma2 model from the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        dtype (torch.dtype): The data type for the model.
        device (Union[str, torch.device]): The device to load the model on.
        disable_mmap (bool, optional): Whether to disable mmap. Defaults to False.
        state_dict (Optional[dict], optional): The state dict to load. Defaults to None.

    Returns:
        gemma2 (Gemma2Model): The loaded model
    """
    logger.info("Building Gemma2")
    GEMMA2_CONFIG = {
      "_name_or_path": "google/gemma-2-2b",
      "architectures": [
        "Gemma2Model"
      ],
      "attention_bias": False,
      "attention_dropout": 0.0,
      "attn_logit_softcapping": 50.0,
      "bos_token_id": 2,
      "cache_implementation": "hybrid",
      "eos_token_id": 1,
      "final_logit_softcapping": 30.0,
      "head_dim": 256,
      "hidden_act": "gelu_pytorch_tanh",
      "hidden_activation": "gelu_pytorch_tanh",
      "hidden_size": 2304,
      "initializer_range": 0.02,
      "intermediate_size": 9216,
      "max_position_embeddings": 8192,
      "model_type": "gemma2",
      "num_attention_heads": 8,
      "num_hidden_layers": 26,
      "num_key_value_heads": 4,
      "pad_token_id": 0,
      "query_pre_attn_scalar": 256,
      "rms_norm_eps": 1e-06,
      "rope_theta": 10000.0,
      "sliding_window": 4096,
      "torch_dtype": "float32",
      "transformers_version": "4.44.2",
      "use_cache": True,
      "vocab_size": 256000
    }

    config = Gemma2Config(**GEMMA2_CONFIG)
    with init_empty_weights():
        gemma2 = Gemma2Model._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(
            ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype
        )

    for key in list(sd.keys()):
        new_key = key.replace("model.", "")
        if new_key == key:
            break  # the model doesn't have annoying prefix
        sd[new_key] = sd.pop(key)

    info = gemma2.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded Gemma2: {info}")
    return gemma2

def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x

DIFFUSERS_TO_ALPHA_VLLM_MAP = {
    # Embedding layers
    "cap_embedder.0.weight": ["time_caption_embed.caption_embedder.0.weight"],
    "cap_embedder.1.weight": "time_caption_embed.caption_embedder.1.weight",
    "cap_embedder.1.bias": "text_embedder.1.bias",
    "x_embedder.weight": "patch_embedder.proj.weight",
    "x_embedder.bias": "patch_embedder.proj.bias",
    # Attention modulation
    "layers.().adaLN_modulation.1.weight": "transformer_blocks.().adaln_modulation.1.weight",
    "layers.().adaLN_modulation.1.bias": "transformer_blocks.().adaln_modulation.1.bias",
    # Final layers
    "final_layer.adaLN_modulation.1.weight": "final_adaln_modulation.1.weight",
    "final_layer.adaLN_modulation.1.bias": "final_adaln_modulation.1.bias",
    "final_layer.linear.weight": "final_linear.weight",
    "final_layer.linear.bias": "final_linear.bias",
    # Noise refiner
    "noise_refiner.().adaLN_modulation.1.weight": "single_transformer_blocks.().adaln_modulation.1.weight",
    "noise_refiner.().adaLN_modulation.1.bias": "single_transformer_blocks.().adaln_modulation.1.bias",
    "noise_refiner.().attention.qkv.weight": "single_transformer_blocks.().attn.to_qkv.weight",
    "noise_refiner.().attention.out.weight": "single_transformer_blocks.().attn.to_out.0.weight",
    # Time embedding
    "t_embedder.mlp.0.weight": "time_embedder.0.weight",
    "t_embedder.mlp.0.bias": "time_embedder.0.bias",
    "t_embedder.mlp.2.weight": "time_embedder.2.weight",
    "t_embedder.mlp.2.bias": "time_embedder.2.bias",
    # Context attention
    "context_refiner.().attention.qkv.weight": "transformer_blocks.().attn2.to_qkv.weight",
    "context_refiner.().attention.out.weight": "transformer_blocks.().attn2.to_out.0.weight",
    # Normalization
    "layers.().attention_norm1.weight": "transformer_blocks.().norm1.weight",
    "layers.().attention_norm2.weight": "transformer_blocks.().norm2.weight",
    # FFN
    "layers.().feed_forward.w1.weight": "transformer_blocks.().ff.net.0.proj.weight",
    "layers.().feed_forward.w2.weight": "transformer_blocks.().ff.net.2.weight",
    "layers.().feed_forward.w3.weight": "transformer_blocks.().ff.net.4.weight",
}


def convert_diffusers_sd_to_alpha_vllm(sd: dict, num_double_blocks: int) -> dict:
    """Convert Diffusers checkpoint to Alpha-VLLM format"""
    logger.info("Converting Diffusers checkpoint to Alpha-VLLM format")
    new_sd = {}

    for key, value in sd.items():
        new_key = key
        for pattern, replacement in DIFFUSERS_TO_ALPHA_VLLM_MAP.items():
            if "()." in pattern:
                for block_idx in range(num_double_blocks):
                    if str(block_idx) in key:
                        converted = pattern.replace("()", str(block_idx))
                        new_key = key.replace(
                            converted, replacement.replace("()", str(block_idx))
                        )
                        break

        if new_key == key:
            logger.debug(f"Unmatched key in conversion: {key}")
        new_sd[new_key] = value

    logger.info(f"Converted {len(new_sd)} keys to Alpha-VLLM format")
    return new_sd
