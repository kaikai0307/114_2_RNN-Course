from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from Load import load_model


def prepare_model():
    # Freeze the base model, including the CLIP vision tower.
    model, processor = load_model()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        exclude_modules=["vision_tower"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    for name, parameter in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            parameter.requires_grad = False
    model.print_trainable_parameters()
    
    return model, processor
