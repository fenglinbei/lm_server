import multiprocessing
import os
from typing import Optional, Dict, List, Union

import dotenv
from loguru import logger
from pydantic import BaseModel, Field

from api.utils.compat import model_json, disable_warnings

dotenv.load_dotenv()

disable_warnings(BaseModel)


def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"


def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default


class Settings(BaseModel):
    """ Settings class. """

    host: Optional[str] = Field(
        default=get_env("HOST", "0.0.0.0"),
        description="Listen address.",
    )
    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),
        description="Listen port.",
    )
    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v2"),
        description="API prefix.",
    )
    api_title: Optional[str] = Field(
        default=get_env("API_TITLE", "LLM_SERVER"),
        description="The title of API."
    )
    engine: Optional[str] = Field(
        default=get_env("ENGINE", "'chatglm.cpp'"),
        description="Choices are ['default', 'vllm', 'llama.cpp', 'chatglm.cpp'].",
    )
    log_path: Optional[str] = Field(
        default=get_env("LOG_PATH", "./api/log/"),
        description="The path where server saved."
    )

    # model related
    model_name: Optional[str] = Field(
        default=get_env("MODEL_NAME", "chatglm3-6b-f16-ggml"),
        description="The name of the model to use for generating completions."
    )
    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", "checkpoints/chatglm3-6b/chatglm3-6b-f16-ggml.bin"),
        description="The path to the model to use for generating completions."
    )
    adapter_model_path: Optional[str] = Field(
        default=get_env("ADAPTER_MODEL_PATH", None),
        description="Path to a LoRA file to apply to the model."
    )
    resize_embeddings: Optional[bool] = Field(
        default=get_bool_env("RESIZE_EMBEDDINGS"),
        description="Whether to resize embeddings."
    )
    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "f16"),
        description="Precision dtype."
    )

    # device related
    device: Optional[str] = Field(
        default=get_env("DEVICE", "cuda"),
        description="Device to load the model."
    )
    device_map: Optional[Union[str, Dict]] = Field(
        default=get_env("DEVICE_MAP", None),
        description="Device map to load the model."
    )
    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),
        description="Specify which gpus to load the model."
    )
    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),
        ge=0,
        description="How many gpus to load the model."
    )

    # embedding related
    only_embedding: Optional[bool] = Field(
        default=get_bool_env("ONLY_EMBEDDING"),
        description="Whether to launch embedding server only."
    )
    embedding_name: Optional[str] = Field(
        default=get_env("EMBEDDING_NAME", "checkpoints/bce-embedding-base_v1/"),
        description="The path to the model to use for generating embeddings."
    )
    embedding_size: Optional[int] = Field(
        default=int(get_env("EMBEDDING_SIZE", -1)),
        description="The embedding size to use for generating embeddings."
    )
    embedding_device: Optional[str] = Field(
        default=get_env("EMBEDDING_DEVICE", "cuda"),
        description="Device to load the model."
    )
    embedding_engine: Optional[str] = Field(
        default=get_env("EMBEDDING_ENGINE", "st"),
        description="The embedding engine."
    )
    triton_port: Optional[int] = Field(
        default=int(get_env("TRITON_PORT", 10001)),
        description="The embedding grpc port while using triton engine."
    )

    # rerank related
    reranker_name: Optional[str] = Field(
        default=get_env("RERANKER_NAME", "checkpoints/bce-reranker-base_v1/"),
        description="The path to the model to use for rerank."
    )

    # quantize related
    quantize: Optional[int] = Field(
        default=int(get_env("QUANTIZE", 16)),
        description="Quantize level for model."
    )
    load_in_8bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_8BIT"),
        description="Whether to load the model in 8 bit."
    )
    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),
        description="Whether to load the model in 4 bit."
    )
    using_ptuning_v2: Optional[bool] = Field(
        default=get_bool_env("USING_PTUNING_V2"),
        description="Whether to load the model using ptuning_v2."
    )
    pre_seq_len: Optional[int] = Field(
        default=int(get_env("PRE_SEQ_LEN", 128)),
        ge=0,
        description="PRE_SEQ_LEN for ptuning_v2."
    )

    # context related
    context_length: Optional[int] = Field(
        default=int(get_env("CONTEXT_LEN", 8192)),
        ge=-1,
        description="Context length for generating completions."
    )
    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),
        description="Chat template for generating completions."
    )
    patch_type: Optional[str] = Field(
        default=get_env("PATCH_TYPE", None),
        description="Patch type for generating completions."
    )
    alpha: Optional[Union[str, float]] = Field(
        default=get_env("ALPHA", "auto"),
        description="Alpha for generating completions."
    )

    # vllm related
    trust_remote_code: Optional[bool] = Field(
        default=get_bool_env("TRUST_REMOTE_CODE"),
        description="Whether to use remote code."
    )
    tokenize_mode: Optional[str] = Field(
        default=get_env("TOKENIZE_MODE", "auto"),
        description="Tokenize mode for vllm server."
    )
    tensor_parallel_size: Optional[int] = Field(
        default=int(get_env("TENSOR_PARALLEL_SIZE", 1)),
        ge=1,
        description="Tensor parallel size for vllm server."
    )
    gpu_memory_utilization: Optional[float] = Field(
        default=float(get_env("GPU_MEMORY_UTILIZATION", 0.9)),
        description="GPU memory utilization for vllm server."
    )
    max_num_batched_tokens: Optional[int] = Field(
        default=int(get_env("MAX_NUM_BATCHED_TOKENS", -1)),
        ge=-1,
        description="Max num batched tokens for vllm server."
    )
    max_num_seqs: Optional[int] = Field(
        default=int(get_env("MAX_NUM_SEQS", 256)),
        ge=1,
        description="Max num seqs for vllm server."
    )
    quantization_method: Optional[str] = Field(
        default=get_env("QUANTIZATION_METHOD", None),
        description="Quantization method for vllm server."
    )

    # support for transformers.TextIteratorStreamer
    use_streamer_v2: Optional[bool] = Field(
        default=get_bool_env("USE_STREAMER_V2"),
        description="Support for transformers.TextIteratorStreamer."
    )

    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,
        description="Support for api key check."
    )

    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),
        description="Whether to activate inference."
    )
    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),
        description="Whether to interrupt requests when a new request is received.",
    )

    # support for llama.cpp
    n_gpu_layers: Optional[int] = Field(
        default=int(get_env("N_GPU_LAYERS", 0)),
        ge=-1,
        description="The number of layers to put on the GPU. The rest will be on the CPU. Set -1 to move all to GPU.",
    )
    main_gpu: Optional[int] = Field(
        default=int(get_env("MAIN_GPU", 0)),
        ge=0,
        description="Main GPU to use.",
    )
    tensor_split: Optional[List[float]] = Field(
        default=float(get_env("TENSOR_SPLIT", None)) if get_env("TENSOR_SPLIT", None) else None,
        description="Split layers across multiple GPUs in proportion.",
    )
    n_batch: Optional[int] = Field(
        default=int(get_env("N_BATCH", 512)),
        ge=1,
        description="The batch size to use per eval."
    )
    n_threads: Optional[int] = Field(
        default=int(get_env("N_THREADS", max(multiprocessing.cpu_count() // 2, 1))),
        ge=1,
        description="The number of threads to use.",
    )
    n_threads_batch: Optional[int] = Field(
        default=int(get_env("N_THREADS_BATCH", max(multiprocessing.cpu_count() // 2, 1))),
        ge=0,
        description="The number of threads to use when batch processing.",
    )
    rope_scaling_type: Optional[int] = Field(
        default=int(get_env("ROPE_SCALING_TYPE", -1))
    )
    rope_freq_base: Optional[float] = Field(
        default=float(get_env("ROPE_FREQ_BASE", 0.0)),
        description="RoPE base frequency"
    )
    rope_freq_scale: Optional[float] = Field(
        default=float(get_env("ROPE_FREQ_SCALE", 0.0)),
        description="RoPE frequency scaling factor",
    )

    # support for tgi
    tgi_endpoint: Optional[str] = Field(
        default=get_env("TGI_ENDPOINT", None),
        description="Text Generate Inference Endpoint.",
    )


SETTINGS = Settings()
logger.debug(f"SETTINGS: {model_json(SETTINGS, indent=4)}")
if SETTINGS.gpus:
    if len(SETTINGS.gpus.split(",")) < SETTINGS.num_gpus:
        raise ValueError(
            f"Larger --num_gpus ({SETTINGS.num_gpus}) than --gpus {SETTINGS.gpus}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS.gpus