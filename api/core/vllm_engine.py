import uuid
import time
import json
import torch
import asyncio
import traceback
from typing import (
    Optional,
    List,
    Dict,
    Any,
    AsyncIterator,
    Union,
    Iterator
)
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.completion import Completion
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from openai.types.chat import ChatCompletionMessageParam
from transformers import PreTrainedTokenizer
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

from api.utils.compat import model_parse
from api.adapter import get_prompt_adapter
from api.generation import build_qwen_chat_input
from api.utils.request import create_error_response
from api.utils.constants import ErrorCode

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)


class VllmEngine:
    def __init__(
        self,
        model: AsyncLLMEngine,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        prompt_name: Optional[str] = None,
        context_len: Optional[int] = -1,
    ):
        """
        Initializes the VLLMEngine object.

        Args:
            model: The AsyncLLMEngine object.
            tokenizer: The PreTrainedTokenizer object.
            model_name: The name of the model.
            prompt_name: The name of the prompt (optional).
            context_len: The length of the context (optional, default=-1).
        """
        self.model = model
        self.model_name = model_name.lower()
        self.tokenizer = tokenizer
        self.prompt_name = prompt_name.lower() if prompt_name is not None else None
        self.prompt_adapter = get_prompt_adapter(self.model_name, prompt_name=self.prompt_name)

        model_config = asyncio.run(self.model.get_model_config())
        if "qwen" in self.model_name:
            self.max_model_len = context_len if context_len > 0 else 8192
        else:
            self.max_model_len = model_config.max_model_len

    def apply_chat_template(
        self,
        messages: List[ChatCompletionMessageParam],
        max_tokens: Optional[int] = 256,
        functions: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False
    ) -> str:
        """
        Applies a chat template to the given messages and returns the processed output.

        Args:
            messages: A list of ChatCompletionMessageParam objects representing the chat messages.
            max_tokens: The maximum number of tokens in the output (optional, default=256).
            functions: A dictionary or list of dictionaries representing the functions to be applied (optional).
            tools: A list of dictionaries representing the tools to be used (optional).

        Returns:
            Union[str, List[int]]: The processed output as a string or a list of integers.
        """
        if self.prompt_adapter.function_call_available:
            messages = self.prompt_adapter.postprocess_messages(
                messages, functions, tools,
            )
            if functions or tools:
                logger.debug(f"==== Messages with tools ====\n{messages}")

        if "chatglm3" in self.model_name:
            query, role = messages[-1]["content"], messages[-1]["role"]
            return self.tokenizer.build_chat_input(
                query, history=messages[:-1], role=role
            )["input_ids"][0].tolist()
        elif "qwen3" in self.model_name:
            prompt: str = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,  # Setting enable_thinking=False disables thinking mode
                    )
            return prompt
        elif "qwen" in self.model_name:
            return build_qwen_chat_input(
                self.tokenizer,
                messages,
                self.max_model_len,
                max_tokens,
                functions,
                tools,
            )
        else:
            return self.prompt_adapter.apply_chat_template(messages)

    def convert_to_inputs(
        self,
        prompt: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        max_tokens: Optional[int] = 256,
    ) -> List[int]:
        max_input_tokens = self.max_model_len - max_tokens
        input_ids = token_ids or self.tokenizer(prompt).input_ids
        return input_ids[-max_input_tokens:]

    async def _generate(self, params: Dict[str, Any], request_id: str) -> AsyncIterator[Union[list[dict], dict]]:
        """
        Generates text based on the given parameters and request ID.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for text generation.
            request_id (str): The ID of the request.

        Yields:
            Any: The generated text.
        """
        max_tokens = params.get("max_tokens", 256)
        prompt_or_messages = params.get("prompt_or_messages")
        prompt = ""
        if isinstance(prompt_or_messages, list):
            prompt: str = self.apply_chat_template(
                prompt_or_messages,
                max_tokens,
                functions=params.get("functions"),
                tools=params.get("tools"),
                enable_thinking=params.get("enable_thinking")
            )
        try:
            sampling_params = SamplingParams(
                n=params.get("n", 1),
                presence_penalty=params.get("presence_penalty", 0.),
                frequency_penalty=params.get("frequency_penalty", 0.),
                repetition_penalty=params.get("repetition_penalty", 1.03),
                temperature=params.get("temperature", 0.9),
                top_p=params.get("top_p", 0.8),
                top_k=params.get("top_k", 20),
                stop=params.get("stop", []),
                stop_token_ids=params.get("stop_token_ids", []),
                max_tokens=params.get("max_tokens", 256),
                min_p=params.get("min_p", 0.0),
                # best_of=params.get("best_of", 1),
                logit_bias=params.get("logit_bias", None),
                logprobs=params.get("logprobs", None),
                seed=params.get("seed", None),
                ignore_eos=params.get("ignore_eos", False),
                skip_special_tokens=params.get("skip_special_tokens", True),
                spaces_between_special_tokens=params.get("spaces_between_special_tokens", True),
            )

            logger.debug(sampling_params)
            results_generator = self.model.generate(
                prompt,
                sampling_params,
                request_id
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        i = 0
        prompt_tokens = 0
        completion_tokens = 0
        results = []
        try:
            async for request_output in results_generator:
                
                if i == 0:
                    prompt_tokens = len(request_output.prompt_token_ids)
                i += 1

                for output in request_output.outputs:
                    completion_tokens = len(output.token_ids)
                    results.append({
                    "index": output.index,
                    "text": output.text,
                    "finish_reason": output.finish_reason,
                    "error_code": 0,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }
                })

                yield results


        except torch.cuda.OutOfMemoryError as e:
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.CUDA_OUT_OF_MEMORY,
            }

        except (ValueError, RuntimeError) as e:
            traceback.print_exc()
            yield {
                "text": f"{server_error_msg}\n\n({e})",
                "error_code": ErrorCode.INTERNAL_ERROR,
            }

    @property
    def stop(self):
        """
        Gets the stop property of the prompt adapter.

        Returns:
            The stop property of the prompt adapter, or None if it does not exist.
        """
        return self.prompt_adapter.stop if hasattr(self.prompt_adapter, "stop") else None
    
    async def _create_chat_completion(self, params: Dict[str, Any]) -> Union[ChatCompletion, JSONResponse]:
        """
        Creates a chat completion based on the given parameters.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.

        Returns:
            ChatCompletion: The generated chat completion.
        """

        chat_id = uuid.uuid4()
        created = int(time.time())
        results_generator = self._generate(params, request_id=chat_id.hex)

        last_outputs = []
        try:
            async for request_output in results_generator:
                last_outputs = request_output
        except asyncio.CancelledError:
            return create_error_response(code=499, message="Cancelled")
        
        if isinstance(last_outputs, dict) and  last_outputs["error_code"] != 0:
            return create_error_response(last_outputs["error_code"], last_outputs["text"])
        
        choices = []
        
        prompt_tokens = last_outputs[0]["usage"]["prompt_tokens"]
        completion_tokens = 0

        for output in last_outputs:

            message = ChatCompletionMessage(
                role="assistant",
                content=output["text"].strip(),
            )

            choice = Choice(
                index=output["index"],
                message=message,
                finish_reason=output["finish_reason"] if output["finish_reason"] else "stop",
            )

            choices.append(choice)
            completion_tokens += output["usage"]["completion_tokens"]
        
        usage = model_parse(CompletionUsage, {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    })

        return ChatCompletion(
            id=f"chat{chat_id}",
            choices=choices,
            created=created,
            model=self.model_name,
            object="chat.completion",
            usage=usage, # type: ignore
        )
    
    async def _create_chat_completion_stream(self, params: Dict[str, Any]) -> AsyncIterator:
        """
        Creates a chat completion stream.

        Args:
            params (Dict[str, Any]): The parameters for generating the chat completion.

        Yields:
            Dict[str, Any]: The output of the chat completion stream.
        """
        
        created = int(time.time())
        chat_id = uuid.uuid4()
        has_function_call = False
        async for output in self._generate(params, request_id=chat_id.hex):
            if output["error_code"] != 0:
                yield output
                return

            finish_reason = output["finish_reason"]
            if len(output["delta"]) == 0 and finish_reason != "function_call":
                continue

            delta = ChoiceDelta(content=output["delta"])

            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason
            )
            yield ChatCompletionChunk(
                id=f"chat{chat_id}",
                choices=[choice],
                created=created,
                model=self.model_name,
                object="chat.completion.chunk",
            )

        if not has_function_call:
            choice = ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason="stop"
            )
            yield ChatCompletionChunk(
                id=f"chat{chat_id}",
                choices=[choice],
                created=created,
                model=self.model_name,
                object="chat.completion.chunk",
            )

    async def create_chat_completion(
        self,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[AsyncIterator, ChatCompletion]:
        params = params or {}
        params |= kwargs
        return (
            self._create_chat_completion_stream(params)
            if params.get("stream", False)
            else self._create_chat_completion(params)
        )