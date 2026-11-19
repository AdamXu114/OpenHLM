#!/usr/bin/env python3
"""Memory-safe conversion of the pi05 JAX checkpoint to PyTorch.

The stock examples/convert_jax_model_to_pytorch.py keeps the full float32 JAX
parameter tree (12.4 GB) alive while building a float32 torch model (13.2 GB),
which needs >28 GB RAM and OOM-killed this machine. This variant:

* frees the JAX parameter tree right after slicing it,
* builds the torch model directly on the GPU in bfloat16,
* drops the host-side state dict after loading it.

Peak host RAM is roughly one copy of the checkpoint (~12.5 GB).
"""

import gc
import importlib.util
import os
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path("/home/xjf/workspace/OpenHLM/src/openpi4OpenHLM")
CKPT = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_base")
OUT = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch")
CONFIG_NAME = "pi05_aloha"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "convert_jax_model_to_pytorch", REPO / "examples/convert_jax_model_to_pytorch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class PaliGemmaConfig:
    def __init__(self):
        self.vision_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "intermediate_size": 4304,
                "patch_size": 14,
                "projection_dim": 2048,
            },
        )()
        self.text_config = type(
            "obj",
            (object,),
            {
                "hidden_size": 2048,
                "num_hidden_layers": 18,
                "num_attention_heads": 8,
                "head_dim": 256,
                "intermediate_size": 16384,
            },
        )()


def main():
    import json
    import shutil

    import openpi.models.gemma
    import openpi.models_pytorch.pi0_pytorch
    import safetensors.torch
    from openpi.training import config as _config

    conv = load_module()
    model_config = _config.get_config(CONFIG_NAME).model

    print(f"Converting {CKPT} -> {OUT} (low-memory mode)", flush=True)
    initial_params = conv.slice_initial_orbax_checkpoint(checkpoint_dir=CKPT, restore_precision="float32")
    print("  JAX params restored", flush=True)

    # --- projection params -------------------------------------------------
    keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    projection_params = {}
    for key in keys:
        kernel = initial_params["projection_params"][key]["kernel"]
        bias = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel, dict):
            weight, bias = kernel["value"], bias["value"]
        else:
            weight, bias = kernel, bias
        projection_params[f"{key}.weight"] = torch.from_numpy(np.array(weight)).T
        projection_params[f"{key}.bias"] = torch.from_numpy(np.array(bias))

    # --- paligemma ---------------------------------------------------------
    paligemma_params, expert_params = conv.slice_paligemma_state_dict(
        initial_params["paligemma_params"], PaliGemmaConfig()
    )
    del initial_params
    gc.collect()
    print("  paligemma sliced, JAX tree freed", flush=True)

    # --- action expert -----------------------------------------------------
    gemma_params = conv.slice_gemma_state_dict(
        expert_params,
        openpi.models.gemma.get_config("gemma_300m"),
        num_expert=1,
        checkpoint_dir=CKPT,
        pi05=model_config.pi05,
    )
    del expert_params
    gc.collect()
    print("  action expert sliced", flush=True)

    all_params = {**paligemma_params, **gemma_params, **projection_params}
    del paligemma_params, gemma_params, projection_params
    gc.collect()

    # Instantiate the model directly in bfloat16 on the GPU: constructing it in
    # float32 on the host would transiently need another 13 GB of RAM.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)
    if device == "cuda":
        torch.set_default_device("cuda")
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    print(f"  model instantiated on {device} ({next(pi0_model.parameters()).dtype})", flush=True)
    pi0_model.load_state_dict(all_params, strict=False)
    del all_params
    gc.collect()
    print("  weights loaded, host state dict freed", flush=True)

    # safetensors writes CPU tensors; move the (bf16) model back before saving.
    torch.set_default_device("cpu")
    pi0_model = pi0_model.to("cpu")
    gc.collect()
    os.makedirs(OUT, exist_ok=True)
    safetensors.torch.save_model(pi0_model, os.path.join(OUT, "model.safetensors"))
    print("  model.safetensors written", flush=True)

    assets_source = pathlib.Path(CKPT).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(OUT) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": "bfloat16",
    }
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print("Conversion completed ->", OUT, flush=True)


if __name__ == "__main__":
    main()
