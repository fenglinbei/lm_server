from __future__ import annotations

import os
import sys
import uuid
import time
import multiprocessing
from typing import (
    List,
    Optional,
    Union,
    Generator,
    Sequence,
    Iterator,
    Deque,
    Callable,
)
from collections import deque

import ctypes

from .llama_types import *
from .llama_grammar import LlamaGrammar
from .llama_cache import (
    BaseLlamaCache,
    LlamaCache,  # type: ignore
    LlamaDiskCache,  # type: ignore
    LlamaRAMCache,  # type: ignore
)
import llama_cpp.llama_cpp as llama_cpp
import llama_cpp.llama_chat_format as llama_chat_format

import numpy as np
import numpy.typing as npt

from ._utils import suppress_stdout_stderr
from ._internals import (
    _LlamaModel,  # type: ignore
    _LlamaContext,  # type: ignore
    _LlamaBatch,  # type: ignore
    _LlamaTokenDataArray,  # type: ignore
)


class Llama:
    """High-level Python wrapper for a llama.cpp model."""

    __backend_initialized = False

    def __init__(
        self,
        model_path: str,
        *,
        # Model Params
        n_gpu_layers: int = 0,
        split_mode: int = llama_cpp.LLAMA_SPLIT_LAYER,
        main_gpu: int = 0,
        tensor_split: Optional[List[float]] = None,
        vocab_only: bool = False,
        use_mmap: bool = True,
        use_mlock: bool = False,
        kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None,
        # Context Params
        seed: int = llama_cpp.LLAMA_DEFAULT_SEED,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        rope_scaling_type: Optional[int] = llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED,
        rope_freq_base: float = 0.0,
        rope_freq_scale: float = 0.0,
        yarn_ext_factor: float = -1.0,
        yarn_attn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_orig_ctx: int = 0,
        mul_mat_q: bool = True,
        logits_all: bool = False,
        embedding: bool = False,
        offload_kqv: bool = False,
        # Sampling Params
        last_n_tokens_size: int = 64,
        # LoRA Params
        lora_base: Optional[str] = None,
        lora_scale: float = 1.0,
        lora_path: Optional[str] = None,
        # Backend Params
        numa: bool = False,
        # Chat Format Params
        chat_format: str = "llama-2",
        chat_handler: Optional[llama_chat_format.LlamaChatCompletionHandler] = None,
        # Misc
        verbose: bool = True,
        # Extra Params
        **kwargs,  # type: ignore
    ):
        """Load a llama.cpp model from `model_path`.

        Examples:
            Basic usage

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ... )
            >>> print(model("The quick brown fox jumps ", stop=["."])["choices"][0]["text"])
            the lazy dog

            Loading a chat model

            >>> import llama_cpp
            >>> model = llama_cpp.Llama(
            ...     model_path="path/to/model",
            ...     chat_format="llama-2",
            ... )
            >>> print(model.create_chat_completion(
            ...     messages=[{
            ...         "role": "user",
            ...         "content": "what is the meaning of life?"
            ...     }]
            ... ))

        Args:
            model_path: Path to the model.
            n_gpu_layers: Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
            split_mode: How to split the model across GPUs. See llama_cpp.LLAMA_SPLIT_* for options.
            main_gpu: main_gpu interpretation depends on split_mode: LLAMA_SPLIT_NONE: the GPU that is used for the entire model. LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results. LLAMA_SPLIT_LAYER: ignored
            tensor_split: How split tensors should be distributed across GPUs. If None, the model is not split.
            vocab_only: Only load the vocabulary no weights.
            use_mmap: Use mmap if possible.
            use_mlock: Force the system to keep the model in RAM.
            kv_overrides: Key-value overrides for the model.
            seed: RNG seed, -1 for random
            n_ctx: Text context, 0 = from model
            n_batch: Prompt processing maximum batch size
            n_threads: Number of threads to use for generation
            n_threads_batch: Number of threads to use for batch processing
            rope_scaling_type: RoPE scaling type, from `enum llama_rope_scaling_type`. ref: https://github.com/ggerganov/llama.cpp/pull/2054
            rope_freq_base: RoPE base frequency, 0 = from model
            rope_freq_scale: RoPE frequency scaling factor, 0 = from model
            yarn_ext_factor: YaRN extrapolation mix factor, negative = from model
            yarn_attn_factor: YaRN magnitude scaling factor
            yarn_beta_fast: YaRN low correction dim
            yarn_beta_slow: YaRN high correction dim
            yarn_orig_ctx: YaRN original context size
            logits_all: Return logits for all tokens, not just the last token. Must be True for completion to return logprobs.
            embedding: Embedding mode only.
            offload_kqv: Offload K, Q, V to GPU.
            last_n_tokens_size: Maximum number of tokens to keep in the last_n_tokens deque.
            lora_base: Optional path to base model, useful if using a quantized base model and you want to apply LoRA to an f16 model.
            lora_path: Path to a LoRA file to apply to the model.
            numa: Enable NUMA support. (NOTE: The initial value of this parameter is used for the remainder of the program as this value is set in llama_backend_init)
            chat_format: String specifying the chat format to use when calling create_chat_completion.
            chat_handler: Optional chat handler to use when calling create_chat_completion.
            verbose: Print verbose output to stderr.

        Raises:
            ValueError: If the model path does not exist.

        Returns:
            A Llama instance.
        """
        self.verbose = verbose

        self.numa = numa
        if not Llama.__backend_initialized:
            with suppress_stdout_stderr(disable=self.verbose):
                llama_cpp.llama_backend_init(self.numa)
            Llama.__backend_initialized = True

        self.model_path = model_path

        # Model Params
        self.model_params = llama_cpp.llama_model_default_params()
        self.model_params.n_gpu_layers = (
            0x7FFFFFFF if n_gpu_layers == -1 else n_gpu_layers
        )  # 0x7FFFFFFF is INT32 max, will be auto set to all layers
        self.model_params.split_mode = split_mode
        self.model_params.main_gpu = main_gpu
        self.tensor_split = tensor_split
        self._c_tensor_split = None
        if self.tensor_split is not None:
            if len(self.tensor_split) > llama_cpp.LLAMA_MAX_DEVICES:
                raise ValueError(
                    f"Attempt to split tensors that exceed maximum supported devices. Current LLAMA_MAX_DEVICES={llama_cpp.LLAMA_MAX_DEVICES}"
                )
            # Type conversion and expand the list to the length of LLAMA_MAX_DEVICES
            FloatArray = ctypes.c_float * llama_cpp.LLAMA_MAX_DEVICES
            self._c_tensor_split = FloatArray(
                *tensor_split  # type: ignore
            )  # keep a reference to the array so it is not gc'd
            self.model_params.tensor_split = self._c_tensor_split
        self.model_params.vocab_only = vocab_only
        self.model_params.use_mmap = use_mmap if lora_path is None else False
        self.model_params.use_mlock = use_mlock

        self.kv_overrides = kv_overrides
        if kv_overrides is not None:
            n_overrides = len(kv_overrides)
            self._kv_overrides_array = llama_cpp.llama_model_kv_override * (n_overrides + 1)
            self._kv_overrides_array_keys = []

            for k, v in kv_overrides.items():
                key_buf = ctypes.create_string_buffer(k.encode("utf-8"))
                self._kv_overrides_array_keys.append(key_buf)
                self._kv_overrides_array[i].key = key_buf
                if isinstance(v, int):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_INT
                    self._kv_overrides_array[i].value.int_value = v
                elif isinstance(v, float):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_FLOAT
                    self._kv_overrides_array[i].value.float_value = v
                elif isinstance(v, bool):
                    self._kv_overrides_array[i].tag = llama_cpp.LLAMA_KV_OVERRIDE_BOOL
                    self._kv_overrides_array[i].value.bool_value = v
                else:
                    raise ValueError(f"Unknown value type for {k}: {v}")

            self._kv_overrides_array_sentinel_key = b'\0'

            # null array sentinel
            self._kv_overrides_array[n_overrides].key = self._kv_overrides_array_sentinel_key
            self.model_params.kv_overrides = self._kv_overrides_array

        self.n_batch = min(n_ctx, n_batch)  # ???
        self.n_threads = n_threads or max(multiprocessing.cpu_count() // 2, 1)
        self.n_threads_batch = n_threads_batch or max(
            multiprocessing.cpu_count() // 2, 1
        )
        # Context Params
        self.context_params = llama_cpp.llama_context_default_params()
        self.context_params.seed = seed
        self.context_params.n_ctx = n_ctx
        self.context_params.n_batch = self.n_batch
        self.context_params.n_threads = self.n_threads
        self.context_params.n_threads_batch = self.n_threads_batch
        self.context_params.rope_scaling_type = (
            rope_scaling_type
            if rope_scaling_type is not None
            else llama_cpp.LLAMA_ROPE_SCALING_UNSPECIFIED
        )
        self.context_params.rope_freq_base = (
            rope_freq_base if rope_freq_base != 0.0 else 0
        )
        self.context_params.rope_freq_scale = (
            rope_freq_scale if rope_freq_scale != 0.0 else 0
        )
        self.context_params.yarn_ext_factor = (
            yarn_ext_factor if yarn_ext_factor != 0.0 else 0
        )
        self.context_params.yarn_attn_factor = (
            yarn_attn_factor if yarn_attn_factor != 0.0 else 0
        )
        self.context_params.yarn_beta_fast = (
            yarn_beta_fast if yarn_beta_fast != 0.0 else 0
        )
        self.context_params.yarn_beta_slow = (
            yarn_beta_slow if yarn_beta_slow != 0.0 else 0
        )
        self.context_params.yarn_orig_ctx = yarn_orig_ctx if yarn_orig_ctx != 0 else 0
        self.context_params.mul_mat_q = mul_mat_q
        self.context_params.logits_all = logits_all
        self.context_params.embedding = embedding
        self.context_params.offload_kqv = offload_kqv

        # Sampling Params
        self.last_n_tokens_size = last_n_tokens_size

        self.cache: Optional[BaseLlamaCache] = None

        self.lora_base = lora_base
        self.lora_scale = lora_scale
        self.lora_path = lora_path

        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        self._model = _LlamaModel(
            path_model=self.model_path, params=self.model_params, verbose=self.verbose
        )
        # Set the default value for the context and correct the batch
        if n_ctx == 0:
            n_ctx = self._model.n_ctx_train()
            self.n_batch = min(n_ctx, n_batch)
            self.context_params.n_ctx = self._model.n_ctx_train()
            self.context_params.n_batch = self.n_batch

        self._ctx = _LlamaContext(
            model=self._model,
            params=self.context_params,
            verbose=self.verbose,
        )

        self._batch = _LlamaBatch(
            n_tokens=self.n_batch,
            embd=0,
            n_seq_max=self.context_params.n_ctx,
            verbose=self.verbose,
        )

        if self.lora_path:
            if self._model.apply_lora_from_file(
                self.lora_path,
                self.lora_scale,
                self.lora_base,
                self.n_threads,
            ):
                raise RuntimeError(
                    f"Failed to apply LoRA from lora path: {self.lora_path} to base path: {self.lora_base}"
                )

        if self.verbose:
            print(llama_cpp.llama_print_system_info().decode("utf-8"), file=sys.stderr)

        self.chat_format = chat_format
        self.chat_handler = chat_handler

        self._n_vocab = self.n_vocab()
        self._n_ctx = self.n_ctx()

        self._token_nl = self.token_nl()
        self._token_eos = self.token_eos()

        self._candidates = _LlamaTokenDataArray(n_vocab=self._n_vocab)

        self.n_tokens = 0
        self.input_ids: npt.NDArray[np.intc] = np.ndarray((n_ctx,), dtype=np.intc)
        self.scores: npt.NDArray[np.single] = np.ndarray(
            (n_ctx, self._n_vocab), dtype=np.single
        )

    @property
    def ctx(self) -> llama_cpp.llama_context_p:
        assert self._ctx.ctx is not None
        return self._ctx.ctx

    @property
    def model(self) -> llama_cpp.llama_model_p:
        assert self._model.model is not None
        return self._model.model

    @property
    def _input_ids(self) -> npt.NDArray[np.intc]:
        return self.input_ids[: self.n_tokens]

    @property
    def _scores(self) -> npt.NDArray[np.single]:
        return self.scores[: self.n_tokens, :]

    @property
    def eval_tokens(self) -> Deque[int]:
        return deque(self.input_ids[: self.n_tokens].tolist(), maxlen=self._n_ctx)

    @property
    def eval_logits(self) -> Deque[List[float]]:
        return deque(
            self.scores[: self.n_tokens, :].tolist(),
            maxlen=self._n_ctx if self.context_params.logits_all else 1,
        )

    def tokenize(
        self, text: bytes, add_bos: bool = True, special: bool = False
    ) -> List[int]:
        """Tokenize a string.

        Args:
            text: The utf-8 encoded string to tokenize.

        Raises:
            RuntimeError: If the tokenization failed.

        Returns:
            A list of tokens.
        """
        return self._model.tokenize(text, add_bos, special)

    def detokenize(self, tokens: List[int]) -> bytes:
        """Detokenize a list of tokens.

        Args:
            tokens: The list of tokens to detokenize.

        Returns:
            The detokenized string.
        """
        return self._model.detokenize(tokens)

    def set_cache(self, cache: Optional[BaseLlamaCache]):
        """Set the cache.

        Args:
            cache: The cache to set.
        """
        self.cache = cache

    def set_seed(self, seed: int):
        """Set the random seed.

        Args:
            seed: The random seed.
        """
        assert self._ctx.ctx is not None
        llama_cpp.llama_set_rng_seed(self._ctx.ctx, seed)

    def reset(self):
        """Reset the model state."""
        self.n_tokens = 0

    def eval(self, tokens: Sequence[int]):
        """Evaluate a list of tokens.

        Args:
            tokens: The list of tokens to evaluate.
        """
        assert self._ctx.ctx is not None
        assert self._batch.batch is not None
        self._ctx.kv_cache_seq_rm(-1, self.n_tokens, -1)
        for i in range(0, len(tokens), self.n_batch):
            batch = tokens[i : min(len(tokens), i + self.n_batch)]
            n_past = self.n_tokens
            n_tokens = len(batch)
            self._batch.set_batch(
                batch=batch, n_past=n_past, logits_all=self.context_params.logits_all
            )
            self._ctx.decode(self._batch)
            # Save tokens
            self.input_ids[n_past : n_past + n_tokens] = batch
            # Save logits
            rows = n_tokens
            cols = self._n_vocab
            offset = (
                0 if self.context_params.logits_all else n_tokens - 1
            )  # NOTE: Only save the last token logits if logits_all is False
            self.scores[n_past + offset : n_past + n_tokens, :].reshape(-1)[
                :
            ] = self._ctx.get_logits()[offset * cols : rows * cols]
            # Update n_tokens
            self.n_tokens += n_tokens

    def sample(
        self,
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_eta: float = 0.1,
        mirostat_tau: float = 5.0,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ):
        """Sample a token from the model.

        Args:
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.

        Returns:
            The sampled token.
        """
        assert self._ctx is not None
        assert self.n_tokens > 0
        last_n_tokens_data = [llama_cpp.llama_token(0)] * max(
            0, self.last_n_tokens_size - self.n_tokens
        ) + self._input_ids[-self.last_n_tokens_size :].tolist()
        last_n_tokens_size = len(last_n_tokens_data)
        n_vocab = self._n_vocab
        n_ctx = self._n_ctx
        top_k = n_vocab if top_k <= 0 else top_k
        last_n_tokens_size = n_ctx if last_n_tokens_size < 0 else last_n_tokens_size
        last_n_tokens_data_c = (llama_cpp.llama_token * last_n_tokens_size)(
            *last_n_tokens_data
        )
        logits: npt.NDArray[np.single] = self._scores[-1, :]

        if logits_processor is not None:
            logits[:] = logits_processor(self._input_ids, logits)

        nl_logit = logits[self._token_nl]
        self._candidates.copy_logits(logits)
        self._ctx.sample_repetition_penalties(
            candidates=self._candidates,
            last_tokens_data=last_n_tokens_data_c,
            penalty_last_n=last_n_tokens_size,
            penalty_repeat=repeat_penalty,
            penalty_freq=frequency_penalty,
            penalty_present=presence_penalty,
        )
        if not penalize_nl:
            self._candidates.candidates.data[self._token_nl].logit = llama_cpp.c_float(
                nl_logit
            )

        if grammar is not None:
            self._ctx.sample_grammar(
                candidates=self._candidates,
                grammar=grammar,
            )

        if temp < 0.0:
            self._ctx.sample_softmax(candidates=self._candidates)
            id = self._candidates.candidates.data[0].id
        elif temp == 0.0:
            id = self._ctx.sample_token_greedy(candidates=self._candidates)
        elif mirostat_mode == 1:
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id = self._ctx.sample_token_mirostat(
                candidates=self._candidates,
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=2.0 * mirostat_tau,
                m=100,
            )
        elif mirostat_mode == 2:
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id = self._ctx.sample_token_mirostat_v2(
                candidates=self._candidates,
                tau=mirostat_tau,
                eta=mirostat_eta,
                mu=2.0 * mirostat_tau,
            )
        else:
            self._ctx.sample_top_k(candidates=self._candidates, k=top_k, min_keep=1)
            self._ctx.sample_tail_free(candidates=self._candidates, z=tfs_z, min_keep=1)
            self._ctx.sample_typical(
                candidates=self._candidates, p=typical_p, min_keep=1
            )
            self._ctx.sample_top_p(candidates=self._candidates, p=top_p, min_keep=1)
            self._ctx.sample_min_p(candidates=self._candidates, p=min_p, min_keep=1)
            self._ctx.sample_temp(candidates=self._candidates, temp=temp)
            id = self._ctx.sample_token(candidates=self._candidates)
        if grammar is not None:
            self._ctx.grammar_accept_token(grammar=grammar, token=id)
        return id

    def generate(
        self,
        tokens: Sequence[int],
        top_k: int = 40,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        temp: float = 0.80,
        repeat_penalty: float = 1.1,
        reset: bool = True,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        penalize_nl: bool = True,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        grammar: Optional[LlamaGrammar] = None,
    ) -> Generator[int, Optional[Sequence[int]], None]:
        """Create a generator of tokens from a prompt.

        Examples:
            >>> llama = Llama("models/ggml-7b.bin")
            >>> tokens = llama.tokenize(b"Hello, world!")
            >>> for token in llama.generate(tokens, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1):
            ...     print(llama.detokenize([token]))

        Args:
            tokens: The prompt tokens.
            top_k: The top-k sampling parameter.
            top_p: The top-p sampling parameter.
            temp: The temperature parameter.
            repeat_penalty: The repeat penalty parameter.
            reset: Whether to reset the model state.

        Yields:
            The generated tokens.
        """
        if reset and self.n_tokens > 0:
            longest_prefix = 0
            for a, b in zip(self._input_ids, tokens[:-1]):
                if a == b:
                    longest_prefix += 1
                else:
                    break
            if longest_prefix > 0:
                if self.verbose:
                    print("Llama.generate: prefix-match hit", file=sys.stderr)
                reset = False
                tokens = tokens[longest_prefix:]
                self.n_tokens = longest_prefix

        if reset:
            self.reset()

        if grammar is not None:
            grammar.reset()

        while True:
            self.eval(tokens)
            token = self.sample(
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                typical_p=typical_p,
                temp=temp,
                repeat_penalty=repeat_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                logits_processor=logits_processor,
                grammar=grammar,
                penalize_nl=penalize_nl,
            )
            if stopping_criteria is not None and stopping_criteria(
                self._input_ids, self._scores[-1, :]
            ):
                return
            tokens_or_none = yield token
            tokens = [token]
            if tokens_or_none is not None:
                tokens.extend(tokens_or_none)

    def create_embedding(
        self, input: Union[str, List[str]], model: Optional[str] = None
    ) -> CreateEmbeddingResponse:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            An embedding object.
        """
        assert self._ctx.ctx is not None
        assert self._model.model is not None
        model_name: str = model if model is not None else self.model_path

        if self.context_params.embedding == False:
            raise RuntimeError(
                "Llama model must be created with embedding=True to call this method"
            )

        if self.verbose:
            llama_cpp.llama_reset_timings(self._ctx.ctx)

        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input

        data: List[Embedding] = []
        total_tokens = 0
        for index, input in enumerate(inputs):
            tokens = self.tokenize(input.encode("utf-8"), special=True)
            self.reset()
            self.eval(tokens)
            n_tokens = len(tokens)
            total_tokens += n_tokens
            embedding = llama_cpp.llama_get_embeddings(self._ctx.ctx)[
                : llama_cpp.llama_n_embd(self._model.model)
            ]

            data.append(
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": index,
                }
            )
        if self.verbose:
            llama_cpp.llama_print_timings(self._ctx.ctx)

        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        }

    def embed(self, input: str) -> List[float]:
        """Embed a string.

        Args:
            input: The utf-8 encoded string to embed.

        Returns:
            A list of embeddings
        """
        return list(map(float, self.create_embedding(input)["data"][0]["embedding"]))

    def _create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[
        Iterator[CreateCompletionResponse], Iterator[CreateCompletionStreamResponse]
    ]:
        assert self._ctx is not None
        assert suffix is None or suffix.__class__ is str

        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        # If prompt is empty, initialize completion with BOS token to avoid
        # detokenization including a space at the beginning of the completion
        completion_tokens: List[int] = [] if len(prompt) > 0 else [self.token_bos()]
        # Add blank space to start of prompt to match OG llama tokenizer
        prompt_tokens: List[int] = (
            (
                self.tokenize(prompt.encode("utf-8"), special=True)
                if prompt != ""
                else [self.token_bos()]
            )
            if isinstance(prompt, str)
            else prompt
        )
        text: bytes = b""
        returned_tokens: int = 0
        stop = (
            stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else []
        )
        model_name: str = model if model is not None else self.model_path

        # NOTE: This likely doesn't work correctly for the first token in the prompt
        # because of the extra space added to the start of the prompt_tokens
        if logit_bias is not None:
            logit_bias_map = {int(k): float(v) for k, v in logit_bias.items()}

            def logit_bias_processor(
                input_ids: npt.NDArray[np.intc],
                scores: npt.NDArray[np.single],
            ) -> npt.NDArray[np.single]:
                new_scores = np.copy(
                    scores
                )  # Does it make sense to copy the whole array or can we just overwrite the original one?
                for input_id, score in logit_bias_map.items():
                    new_scores[input_id] = score + scores[input_id]
                return new_scores

            _logit_bias_processor = LogitsProcessorList([logit_bias_processor])
            if logits_processor is None:
                logits_processor = _logit_bias_processor
            else:
                logits_processor = logits_processor.extend(_logit_bias_processor)

        if self.verbose:
            self._ctx.reset_timings()

        if len(prompt_tokens) >= self._n_ctx:
            raise ValueError(
                f"Requested tokens ({len(prompt_tokens)}) exceed context window of {llama_cpp.llama_n_ctx(self.ctx)}"
            )

        if max_tokens is None or max_tokens <= 0:
            # Unlimited, depending on n_ctx.
            max_tokens = self._n_ctx - len(prompt_tokens)

        # Truncate max_tokens if requested tokens would exceed the context window
        max_tokens = (
            max_tokens
            if max_tokens + len(prompt_tokens) < self._n_ctx
            else (self._n_ctx - len(prompt_tokens))
        )

        if stop != []:
            stop_sequences = [s.encode("utf-8") for s in stop]
        else:
            stop_sequences = []

        if logprobs is not None and self.context_params.logits_all is False:
            raise ValueError(
                "logprobs is not supported for models created with logits_all=False"
            )

        if self.cache:
            try:
                cache_item = self.cache[prompt_tokens]
                cache_prefix_len = Llama.longest_token_prefix(
                    cache_item.input_ids.tolist(), prompt_tokens
                )
                eval_prefix_len = Llama.longest_token_prefix(
                    self._input_ids.tolist(), prompt_tokens
                )
                if cache_prefix_len > eval_prefix_len:
                    self.load_state(cache_item)
                    if self.verbose:
                        print("Llama._create_completion: cache hit", file=sys.stderr)
            except KeyError:
                if self.verbose:
                    print("Llama._create_completion: cache miss", file=sys.stderr)

        if seed is not None:
            self._ctx.set_rng_seed(seed)

        finish_reason = "length"
        multibyte_fix = 0
        for token in self.generate(
            prompt_tokens,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            temp=temperature,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
        ):
            if token == self._token_eos:
                text = self.detokenize(completion_tokens)
                finish_reason = "stop"
                break

            completion_tokens.append(token)

            all_text = self.detokenize(completion_tokens)

            # Contains multi-byte UTF8
            for k, char in enumerate(all_text[-3:]):
                k = 3 - k
                for num, pattern in [(2, 192), (3, 224), (4, 240)]:
                    # Bitwise AND check
                    if num > k and pattern & char == pattern:
                        multibyte_fix = num - k

            # Stop incomplete bytes from passing
            if multibyte_fix > 0:
                multibyte_fix -= 1
                continue

            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                first_stop = any_stop[0]
                text = all_text[: all_text.index(first_stop)]
                finish_reason = "stop"
                break

            if stream:
                remaining_tokens = completion_tokens[returned_tokens:]
                remaining_text = self.detokenize(remaining_tokens)
                remaining_length = len(remaining_text)

                # We want to avoid yielding any characters from
                # the generated text if they are part of a stop
                # sequence.
                first_stop_position = 0
                for s in stop_sequences:
                    for i in range(min(len(s), remaining_length), 0, -1):
                        if remaining_text.endswith(s[:i]):
                            if i > first_stop_position:
                                first_stop_position = i
                            break

                token_end_position = 0

                if logprobs is not None:
                    # not sure how to handle this branch when dealing
                    # with CJK output, so keep it unchanged
                    for token in remaining_tokens:
                        if token == self.token_bos():
                            continue
                        token_end_position += len(self.detokenize([token]))
                        # Check if stop sequence is in the token
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        token_str = self.detokenize([token]).decode(
                            "utf-8", errors="ignore"
                        )
                        text_offset = len(prompt) + len(
                            self.detokenize(completion_tokens[:returned_tokens]).decode(
                                "utf-8", errors="ignore"
                            )
                        )
                        token_offset = len(prompt_tokens) + returned_tokens
                        logits = self._scores[token_offset - 1, :]
                        current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                        sorted_logprobs = list(
                            sorted(
                                zip(current_logprobs, range(len(current_logprobs))),
                                reverse=True,
                            )
                        )
                        top_logprob = {
                            self.detokenize([i]).decode(
                                "utf-8", errors="ignore"
                            ): logprob
                            for logprob, i in sorted_logprobs[:logprobs]
                        }
                        top_logprob.update({token_str: current_logprobs[int(token)]})
                        logprobs_or_none = {
                            "tokens": [
                                self.detokenize([token]).decode(
                                    "utf-8", errors="ignore"
                                )
                            ],
                            "text_offset": [text_offset],
                            "token_logprobs": [current_logprobs[int(token)]],
                            "top_logprobs": [top_logprob],
                        }
                        returned_tokens += 1
                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": self.detokenize([token]).decode(
                                        "utf-8", errors="ignore"
                                    ),
                                    "index": 0,
                                    "logprobs": logprobs_or_none,
                                    "finish_reason": None,
                                }
                            ],
                        }
                else:
                    while len(remaining_tokens) > 0:
                        decode_success = False
                        for i in range(1, len(remaining_tokens) + 1):
                            try:
                                bs = self.detokenize(remaining_tokens[:i])
                                ts = bs.decode("utf-8")
                                decode_success = True
                                break
                            except UnicodeError:
                                pass
                        else:
                            break
                        if not decode_success:
                            # all remaining tokens cannot be decoded to a UTF-8 character
                            break
                        token_end_position += len(bs)
                        if token_end_position > (
                            remaining_length - first_stop_position
                        ):
                            break
                        remaining_tokens = remaining_tokens[i:]
                        returned_tokens += i

                        yield {
                            "id": completion_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": ts,
                                    "index": 0,
                                    "logprobs": None,
                                    "finish_reason": None,
                                }
                            ],
                        }

            if len(completion_tokens) >= max_tokens:
                text = self.detokenize(completion_tokens)
                finish_reason = "length"
                break

        if stopping_criteria is not None and stopping_criteria(
            self._input_ids, self._scores[-1, :]
        ):
            text = self.detokenize(completion_tokens)
            finish_reason = "stop"

        if self.verbose:
            self._ctx.print_timings()

        if stream:
            remaining_tokens = completion_tokens[returned_tokens:]
            all_text = self.detokenize(remaining_tokens)
            any_stop = [s for s in stop_sequences if s in all_text]
            if len(any_stop) > 0:
                end = min(all_text.index(stop) for stop in any_stop)
            else:
                end = len(all_text)

            token_end_position = 0
            for token in remaining_tokens:
                token_end_position += len(self.detokenize([token]))

                logprobs_or_none: Optional[CompletionLogprobs] = None
                if logprobs is not None:
                    if token == self.token_bos():
                        continue
                    token_str = self.detokenize([token]).decode(
                        "utf-8", errors="ignore"
                    )
                    text_offset = len(prompt) + len(
                        self.detokenize(completion_tokens[:returned_tokens])
                    )
                    token_offset = len(prompt_tokens) + returned_tokens - 1
                    logits = self._scores[token_offset, :]
                    current_logprobs = Llama.logits_to_logprobs(logits).tolist()
                    sorted_logprobs = list(
                        sorted(
                            zip(current_logprobs, range(len(current_logprobs))),
                            reverse=True,
                        )
                    )
                    top_logprob = {
                        self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                        for logprob, i in sorted_logprobs[:logprobs]
                    }
                    top_logprob.update({token_str: current_logprobs[int(token)]})
                    logprobs_or_none = {
                        "tokens": [
                            self.detokenize([token]).decode("utf-8", errors="ignore")
                        ],
                        "text_offset": [text_offset],
                        "token_logprobs": [current_logprobs[int(token)]],
                        "top_logprobs": [top_logprob],
                    }

                if token_end_position >= end:
                    last_text = self.detokenize([token])
                    if token_end_position == end - 1:
                        break
                    returned_tokens += 1
                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "text": last_text[
                                    : len(last_text) - (token_end_position - end)
                                ].decode("utf-8", errors="ignore"),
                                "index": 0,
                                "logprobs": logprobs_or_none,
                                "finish_reason": None,
                            }
                        ],
                    }
                    break
                returned_tokens += 1
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": self.detokenize([token]).decode(
                                "utf-8", errors="ignore"
                            ),
                            "index": 0,
                            "logprobs": logprobs_or_none,
                            "finish_reason": None,
                        }
                    ],
                }
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "text": "",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if self.cache:
                if self.verbose:
                    print("Llama._create_completion: cache save", file=sys.stderr)
                self.cache[prompt_tokens + completion_tokens] = self.save_state()
                print("Llama._create_completion: cache saved", file=sys.stderr)
            return

        if self.cache:
            if self.verbose:
                print("Llama._create_completion: cache save", file=sys.stderr)
            self.cache[prompt_tokens + completion_tokens] = self.save_state()

        text_str = text.decode("utf-8", errors="ignore")

        if echo:
            text_str = prompt + text_str

        if suffix is not None:
            text_str = text_str + suffix

        logprobs_or_none: Optional[CompletionLogprobs] = None
        if logprobs is not None:
            text_offset = 0 if echo else len(prompt)
            token_offset = 0 if echo else len(prompt_tokens[1:])
            text_offsets: List[int] = []
            token_logprobs: List[Optional[float]] = []
            tokens: List[str] = []
            top_logprobs: List[Optional[Dict[str, float]]] = []

            if echo:
                # Remove leading BOS token
                all_tokens = prompt_tokens[1:] + completion_tokens
            else:
                all_tokens = completion_tokens

            all_token_strs = [
                self.detokenize([token]).decode("utf-8", errors="ignore")
                for token in all_tokens
            ]
            all_logprobs = Llama.logits_to_logprobs(self._scores)[token_offset:]
            # TODO: may be able to change this loop to use np.take_along_dim
            for idx, (token, token_str, logprobs_token) in enumerate(
                zip(all_tokens, all_token_strs, all_logprobs)
            ):
                if token == self.token_bos():
                    continue
                text_offsets.append(
                    text_offset
                    + len(
                        self.detokenize(all_tokens[:idx]).decode(
                            "utf-8", errors="ignore"
                        )
                    )
                )
                tokens.append(token_str)
                sorted_logprobs = list(
                    sorted(
                        zip(logprobs_token, range(len(logprobs_token))), reverse=True
                    )
                )
                token_logprobs.append(logprobs_token[int(token)])
                top_logprob: Optional[Dict[str, float]] = {
                    self.detokenize([i]).decode("utf-8", errors="ignore"): logprob
                    for logprob, i in sorted_logprobs[:logprobs]
                }
                top_logprob.update({token_str: logprobs_token[int(token)]})
                top_logprobs.append(top_logprob)
            # Weird idosincracy of the OpenAI API where
            # token_logprobs and top_logprobs are null for
            # the first token.
            if echo and len(all_tokens) > 0:
                token_logprobs[0] = None
                top_logprobs[0] = None
            logprobs_or_none = {
                "tokens": tokens,
                "text_offset": text_offsets,
                "token_logprobs": token_logprobs,
                "top_logprobs": top_logprobs,
            }

        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "text": text_str,
                    "index": 0,
                    "logprobs": logprobs_or_none,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(completion_tokens),
                "total_tokens": len(prompt_tokens) + len(completion_tokens),
            },
        }

    def create_completion(
        self,
        prompt: Union[str, List[int]],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.
            logit_bias: A logit bias to use.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=-1 if max_tokens is None else max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            seed=seed,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )
        if stream:
            chunks: Iterator[CreateCompletionStreamResponse] = completion_or_chunks
            return chunks
        completion: Completion = next(completion_or_chunks)  # type: ignore
        return completion

    def __call__(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = 16,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 40,
        stream: bool = False,
        seed: Optional[int] = None,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            stream: Whether to stream the results.
            seed: The seed to use for sampling.
            tfs_z: The tail-free sampling parameter. Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
            mirostat_eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
            model: The name to use for the model in the completion object.
            stopping_criteria: A list of stopping criteria to use.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use for constrained sampling.
            logit_bias: A logit bias to use.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        return self.create_completion(
            prompt=prompt,
            suffix=suffix,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            typical_p=typical_p,
            logprobs=logprobs,
            echo=echo,
            stop=stop,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            stream=stream,
            seed=seed,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )

    def create_chat_completion(
        self,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[ChatCompletionFunction]] = None,
        function_call: Optional[ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[ChatCompletionTool]] = None,
        tool_choice: Optional[ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        grammar: Optional[LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
    ) -> Union[
        CreateChatCompletionResponse, Iterator[CreateChatCompletionStreamResponse]
    ]:
        """Generate a chat completion from a list of messages.

        Args:
            messages: A list of messages to generate a response for.
            functions: A list of functions to use for the chat completion.
            function_call: A function call to use for the chat completion.
            tools: A list of tools to use for the chat completion.
            tool_choice: A tool choice to use for the chat completion.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for nucleus sampling. Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            top_k: The top-k value to use for sampling. Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
            min_p: The min-p value to use for minimum p sampling. Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
            typical_p: The typical-p value to use for sampling. Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
            stream: Whether to stream the results.
            stop: A list of strings to stop generation when encountered.
            seed: The seed to use for sampling.
            response_format: The response format to use for the chat completion. Use { "type": "json_object" } to contstrain output to only valid json.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0 or None, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            presence_penalty: The penalty to apply to tokens based on their presence in the prompt.
            frequency_penalty: The penalty to apply to tokens based on their frequency in the prompt.
            repeat_penalty: The penalty to apply to repeated tokens.
            tfs_z: The tail-free sampling parameter.
            mirostat_mode: The mirostat sampling mode.
            mirostat_tau: The mirostat sampling tau parameter.
            mirostat_eta: The mirostat sampling eta parameter.
            model: The name to use for the model in the completion object.
            logits_processor: A list of logits processors to use.
            grammar: A grammar to use.
            logit_bias: A logit bias to use.

        Returns:
            Generated chat completion or a stream of chat completion chunks.
        """
        handler = self.chat_handler or llama_chat_format.get_chat_completion_handler(
            self.chat_format
        )
        return handler(
            llama=self,
            messages=messages,
            functions=functions,
            function_call=function_call,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            typical_p=typical_p,
            stream=stream,
            stop=stop,
            seed=seed,
            response_format=response_format,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repeat_penalty=repeat_penalty,
            tfs_z=tfs_z,
            mirostat_mode=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            model=model,
            logits_processor=logits_processor,
            grammar=grammar,
            logit_bias=logit_bias,
        )

    def __getstate__(self):
        return dict(
            model_path=self.model_path,
            # Model Params
            n_gpu_layers=self.model_params.n_gpu_layers,
            split_mode=self.model_params.split_mode,
            main_gpu=self.model_params.main_gpu,
            tensor_split=self.tensor_split,
            vocab_only=self.model_params.vocab_only,
            use_mmap=self.model_params.use_mmap,
            use_mlock=self.model_params.use_mlock,
            kv_overrides=self.kv_overrides,
            # Context Params
            seed=self.context_params.seed,
            n_ctx=self.context_params.n_ctx,
            n_batch=self.n_batch,
            n_threads=self.context_params.n_threads,
            n_threads_batch=self.context_params.n_threads_batch,
            rope_scaling_type=self.context_params.rope_scaling_type,
            rope_freq_base=self.context_params.rope_freq_base,
            rope_freq_scale=self.context_params.rope_freq_scale,
            yarn_ext_factor=self.context_params.yarn_ext_factor,
            yarn_attn_factor=self.context_params.yarn_attn_factor,
            yarn_beta_fast=self.context_params.yarn_beta_fast,
            yarn_beta_slow=self.context_params.yarn_beta_slow,
            yarn_orig_ctx=self.context_params.yarn_orig_ctx,
            mul_mat_q=self.context_params.mul_mat_q,
            logits_all=self.context_params.logits_all,
            embedding=self.context_params.embedding,
            # Sampling Params
            last_n_tokens_size=self.last_n_tokens_size,
            # LoRA Params
            lora_base=self.lora_base,
            lora_scale=self.lora_scale,
            lora_path=self.lora_path,
            # Backend Params
            numa=self.numa,
            # Chat Format Params
            chat_format=self.chat_format,
            chat_handler=self.chat_handler,
            # Misc
            verbose=self.verbose,
        )

    def __setstate__(self, state):
        self.__init__(
            model_path=state["model_path"],
            # Model Params
            n_gpu_layers=state["n_gpu_layers"],
            split_mode=state["split_mode"],
            main_gpu=state["main_gpu"],
            tensor_split=state["tensor_split"],
            vocab_only=state["vocab_only"],
            use_mmap=state["use_mmap"],
            use_mlock=state["use_mlock"],
            kv_overrides=state["kv_overrides"],
            # Context Params
            seed=state["seed"],
            n_ctx=state["n_ctx"],
            n_batch=state["n_batch"],
            n_threads=state["n_threads"],
            n_threads_batch=state["n_threads_batch"],
            rope_freq_base=state["rope_freq_base"],
            rope_freq_scale=state["rope_freq_scale"],
            rope_scaling_type=state["rope_scaling_type"],
            yarn_ext_factor=state["yarn_ext_factor"],
            yarn_attn_factor=state["yarn_attn_factor"],
            yarn_beta_fast=state["yarn_beta_fast"],
            yarn_beta_slow=state["yarn_beta_slow"],
            yarn_orig_ctx=state["yarn_orig_ctx"],
            mul_mat_q=state["mul_mat_q"],
            logits_all=state["logits_all"],
            embedding=state["embedding"],
            # Sampling Params
            last_n_tokens_size=state["last_n_tokens_size"],
            # LoRA Params
            lora_base=state["lora_base"],
            lora_path=state["lora_path"],
            # Backend Params
            numa=state["numa"],
            # Chat Format Params
            chat_format=state["chat_format"],
            chat_handler=state["chat_handler"],
            # Misc
            verbose=state["verbose"],
        )

    def save_state(self) -> LlamaState:
        assert self._ctx.ctx is not None
        if self.verbose:
            print("Llama.save_state: saving llama state", file=sys.stderr)
        state_size = llama_cpp.llama_get_state_size(self._ctx.ctx)
        if self.verbose:
            print(f"Llama.save_state: got state size: {state_size}", file=sys.stderr)
        llama_state = (llama_cpp.c_uint8 * int(state_size))()
        if self.verbose:
            print("Llama.save_state: allocated state", file=sys.stderr)
        n_bytes = llama_cpp.llama_copy_state_data(self._ctx.ctx, llama_state)
        if self.verbose:
            print(f"Llama.save_state: copied llama state: {n_bytes}", file=sys.stderr)
        if int(n_bytes) > int(state_size):
            raise RuntimeError("Failed to copy llama state data")
        llama_state_compact = (llama_cpp.c_uint8 * int(n_bytes))()
        llama_cpp.ctypes.memmove(llama_state_compact, llama_state, int(n_bytes))
        if self.verbose:
            print(
                f"Llama.save_state: saving {n_bytes} bytes of llama state",
                file=sys.stderr,
            )
        return LlamaState(
            scores=self.scores.copy(),
            input_ids=self.input_ids.copy(),
            n_tokens=self.n_tokens,
            llama_state=bytes(llama_state_compact),
            llama_state_size=n_bytes,
        )

    def load_state(self, state: LlamaState) -> None:
        assert self._ctx.ctx is not None
        self.scores = state.scores.copy()
        self.input_ids = state.input_ids.copy()
        self.n_tokens = state.n_tokens
        state_size = state.llama_state_size
        LLamaStateArrayType = llama_cpp.c_uint8 * state_size
        llama_state = LLamaStateArrayType.from_buffer_copy(state.llama_state)

        if llama_cpp.llama_set_state_data(self._ctx.ctx, llama_state) != state_size:
            raise RuntimeError("Failed to set llama state data")

    def n_ctx(self) -> int:
        """Return the context window size."""
        return self._ctx.n_ctx()

    def n_embd(self) -> int:
        """Return the embedding size."""
        return self._model.n_embd()

    def n_vocab(self) -> int:
        """Return the vocabulary size."""
        return self._model.n_vocab()

    def tokenizer(self) -> "LlamaTokenizer":
        """Return the tokenizer for this model."""
        return LlamaTokenizer(self)

    def token_eos(self) -> int:
        """Return the end-of-sequence token."""
        return self._model.token_eos()

    def token_bos(self) -> int:
        """Return the beginning-of-sequence token."""
        return self._model.token_bos()

    def token_nl(self) -> int:
        """Return the newline token."""
        return self._model.token_nl()

    @staticmethod
    def logits_to_logprobs(
        logits: Union[npt.NDArray[np.single], List], axis: int = -1
    ) -> npt.NDArray[np.single]:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_softmax.html
        logits_maxs: np.ndarray = np.amax(logits, axis=axis, keepdims=True)
        if logits_maxs.ndim > 0:
            logits_maxs[~np.isfinite(logits_maxs)] = 0
        elif not np.isfinite(logits_maxs):
            logits_maxs = 0
        subtract_maxs = np.subtract(logits, logits_maxs, dtype=np.single)
        exp = np.exp(subtract_maxs)
        # Suppress warnings about log of zero
        with np.errstate(divide="ignore"):
            summed = np.sum(exp, axis=axis, keepdims=True)
            out = np.log(summed)
        return subtract_maxs - out

    @staticmethod
    def longest_token_prefix(a: Sequence[int], b: Sequence[int]):
        longest_prefix = 0
        for _a, _b in zip(a, b):
            if _a == _b:
                longest_prefix += 1
            else:
                break
        return longest_prefix


class LlamaTokenizer:
    def __init__(self, llama: Llama):
        self.llama = llama

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        return self.llama.tokenize(
            text.encode("utf-8", errors="ignore"), add_bos=add_bos, special=True
        )

    def decode(self, tokens: List[int]) -> str:
        return self.llama.detokenize(tokens).decode("utf-8", errors="ignore")

    @classmethod
    def from_ggml_file(cls, path: str) -> "LlamaTokenizer":
        return cls(Llama(model_path=path, vocab_only=True))


class LlamaState:
    def __init__(
        self,
        input_ids: npt.NDArray[np.intc],
        scores: npt.NDArray[np.single],
        n_tokens: int,
        llama_state: bytes,
        llama_state_size: int,
    ):
        self.input_ids = input_ids
        self.scores = scores
        self.n_tokens = n_tokens
        self.llama_state = llama_state
        self.llama_state_size = llama_state_size


LogitsProcessor = Callable[
    [npt.NDArray[np.intc], npt.NDArray[np.single]], npt.NDArray[np.single]
]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], scores: npt.NDArray[np.single]
    ) -> npt.NDArray[np.single]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


StoppingCriteria = Callable[[npt.NDArray[np.intc], npt.NDArray[np.single]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(
        self, input_ids: npt.NDArray[np.intc], logits: npt.NDArray[np.single]
    ) -> bool:
        return any([stopping_criteria(input_ids, logits) for stopping_criteria in self])
