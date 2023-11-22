import asyncio

from api.apapter import get_prompt_adapter
from api.config import config
from api.generation import ModelServer


def get_generate_model():
    from api.apapter.model import load_model

    if config.PATCH_TYPE == "rerope":
        from api.utils.patches import apply_rerope_patch

        apply_rerope_patch(config.TRAINING_LENGTH, config.WINDOW_SIZE)
    elif config.PATCH_TYPE == "ntk":
        from api.utils.patches import apply_ntk_scaling_patch

        apply_ntk_scaling_patch(config.TRAINING_LENGTH)

    model, tokenizer = load_model(
        config.MODEL_NAME,
        model_dir=config.MODEL_PATH,
        adapter_model=config.ADAPTER_MODEL_PATH,
        quantize=config.QUANTIZE,
        device=config.DEVICE,
        device_map=config.DEVICE_MAP,
        num_gpus=config.NUM_GPUs,
        load_in_8bit=config.LOAD_IN_8BIT,
        load_in_4bit=config.LOAD_IN_4BIT,
        use_ptuning_v2=config.USING_PTUNING_V2,
    )

    return ModelServer(
        model,
        tokenizer,
        config.DEVICE,
        model_name=config.MODEL_NAME,
        context_len=config.CONTEXT_LEN,
        stream_interval=config.STREAM_INTERVERL,
        prompt_name=config.PROMPT_NAME,
    )


def get_context_len(model_config):
    if "qwen" in config.MODEL_NAME.lower():
        max_model_len = config.CONTEXT_LEN or 8192
    else:
        max_model_len = config.CONTEXT_LEN or model_config.get_max_model_len()
    return max_model_len


# EMBEDDED_MODEL = get_embedding_model() if config.EMBEDDING_NAME else None  # model for embedding
GENERATE_MDDEL = get_generate_model() if not config.USE_VLLM else None  # model for transformers generate
EXCLUDE_MODELS = ["chatglm"]
