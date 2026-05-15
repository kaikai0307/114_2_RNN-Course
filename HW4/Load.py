# pip install torch transformers accelerate bitsandbytes peft datasets trl pillow
from pathlib import Path

import bitsandbytes.functional as bnb_functional
import torch
import torch.nn as nn
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


DEFAULT_BASE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"


def get_multi_modal_projector(model):
    candidate_paths = [
        ("model", "multi_modal_projector"),
        ("base_model", "model", "multi_modal_projector"),
        ("base_model", "model", "model", "multi_modal_projector"),
    ]

    for path in candidate_paths:
        current = model
        found = True
        for attr in path:
            if not hasattr(current, attr):
                found = False
                break
            current = getattr(current, attr)
        if found:
            return current

    raise AttributeError("Could not locate multi_modal_projector on model")


def dequantize_projector_linear(layer):
    weight = bnb_functional.dequantize_4bit(
        layer.weight.data,
        quant_state=layer.weight.quant_state,
        quant_type=layer.weight.quant_type,
    ).to(torch.float32)
    new_layer = nn.Linear(
        layer.in_features,
        layer.out_features,
        bias=layer.bias is not None,
        device=weight.device,
        dtype=torch.float32,
    )
    new_layer.weight.data.copy_(weight)
    if layer.bias is not None:
        new_layer.bias.data.copy_(layer.bias.data.to(torch.float32))
    return new_layer


def prepare_trainable_projector(model):
    projector = get_multi_modal_projector(model)
    if not hasattr(projector.linear_1.weight, "quant_state") and not hasattr(projector.linear_2.weight, "quant_state"):
        return model

    projector.linear_1 = dequantize_projector_linear(projector.linear_1)
    projector.linear_2 = dequantize_projector_linear(projector.linear_2)
    return model


def cast_projector_for_inference(model, dtype=torch.float16):
    get_multi_modal_projector(model).to(dtype=dtype)
    return model


def load_adapter_state(adapter_dir: Path):
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.exists():
        return load_file(str(adapter_file))

    raise FileNotFoundError(f"Adapter weights not found in {adapter_dir}")


def load_lora_adapter(model, adapter_dir: Path):
    state_dict = load_adapter_state(adapter_dir)

    projector_prefix = "base_model.model.model.multi_modal_projector."
    projector_state = {
        key.removeprefix(projector_prefix): value
        for key, value in state_dict.items()
        if key.startswith(projector_prefix)
    }
    if projector_state:
        get_multi_modal_projector(model).load_state_dict(projector_state, strict=True)

    lora_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(projector_prefix)
    }

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    peft_config.modules_to_save = None
    model = get_peft_model(model, peft_config)
    set_peft_model_state_dict(model, lora_state)
    return model


def load_model(
    model_id: str | None = None,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
):
    if model_id is None:
        model_id = DEFAULT_BASE_MODEL_ID

    quantization_config = None
    torch_dtype = torch.float16

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    print("Loading base LLaVA model...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=None if load_in_4bit else torch_dtype,
        device_map="auto",
    )

    processor_source = model_id
    if adapter_path is not None:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        model = prepare_trainable_projector(model)
        print(f"Loading LoRA adapter from {adapter_dir} ...")
        model = load_lora_adapter(model, adapter_dir)
        model = cast_projector_for_inference(model)
        processor_source = str(adapter_dir)

    processor = AutoProcessor.from_pretrained(processor_source)
    print("Model loaded.")
    return model, processor
