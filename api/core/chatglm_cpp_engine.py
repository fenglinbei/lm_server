import time
import uuid
from typing import (
    Optional,
    Tuple,
    List,
    Union,
    Dict,
    Iterator,
    Any,
)

from chatglm_cpp import Pipeline, _C
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
)
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from api.adapter import get_prompt_adapter
from api.utils.compat import model_parse

from loguru import logger

def apply_stopping_strings(reply: str, stop_strings: List[str]) -> Tuple[str, bool]:
    """
    Apply stopping strings to the reply and check if a stop string is found.

    Args:
        reply (str): The reply to apply stopping strings to.
        stop_strings (List[str]): The list of stopping strings to check for.

    Returns:
        Tuple[str, bool]: A tuple containing the modified reply and a boolean indicating if a stop string was found.
    """
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found


class ChatglmCppEngine:
    def __init__(
        self,
        pipeline: Pipeline,
        model_name: str
    ):
        """
        Initializes a ChatglmCppEngine instance.

        Args:
            pipeline (Pipeline): The chatglm's pipeline to be used by the engine.
            model_name (str): The name of the model.
        """

        self.pipeline = pipeline
        self.model_name = model_name.lower()

    def create_completion(self, input_ids, gen_config) -> str:
        input_echo_len = len(input_ids)
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        output_ids = self.pipeline._sync_generate_ids(input_ids=input_ids, gen_config=gen_config)
        total_len = len(output_ids) + input_echo_len
        response = self.pipeline.tokenizer.decode(output_ids)
        return {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": self.model_name,
                    "delta": response,
                    "text": response,
                    "logprobs": None,
                    "finish_reason": "stop",
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": total_len - input_echo_len,
                        "total_tokens": total_len,
                    },
                }
    
    def create_steam_completion(self, input_ids: List[int], gen_config: _C.GenerationConfig) -> Iterator[dict]:
        input_ids = input_ids.copy()
        n_past = 0
        total_len, previous_text = 0, ""
        completion_id: str = f"cmpl-{str(uuid.uuid4())}"
        created: int = int(time.time())
        output_ids: List[int] = []
        input_echo_len = len(input_ids)
        max_tokens = gen_config.max_length

        while len(input_ids) < max_tokens:
            next_token_id = self.pipeline.model.generate_next_token(input_ids, gen_config, n_past, input_echo_len)
            n_past = len(input_ids)
            input_ids.append(next_token_id)

            output_ids.append(next_token_id)
            total_len = len(output_ids)
            response = self.pipeline.tokenizer.decode(output_ids)

            if response:
                if response.endswith((",", "!", ":", ";", "?", "�")):
                    pass
                else:
                    response, stop_found = apply_stopping_strings(response, ["<|observation|>"])
                    delta_text = response[len(previous_text):]
                    previous_text = response

                    yield {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": self.model_name,
                        "delta": response,
                        "text": delta_text,
                        "logprobs": None,
                        "finish_reason": "function_call" if stop_found else None,
                        "usage": {
                            "prompt_tokens": input_echo_len,
                            "completion_tokens": total_len - input_echo_len,
                            "total_tokens": total_len,
                        },
                    }

                if stop_found:
                    break

            if next_token_id in [self.pipeline.model.config.eos_token_id, *self.pipeline.model.config.extra_eos_token_ids]:
                break
        # Only last stream result contains finish_reason, we set finish_reason as stop
        yield {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": self.model_name,
            "delta": "",
            "text": response,
            "logprobs": None,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
        }

    def _create_chat_completion(self, input_ids, gen_config) -> ChatCompletion:
        completion = self.create_completion(input_ids, gen_config)
        message = ChatCompletionMessage(
            role="assistant",
            content=completion["text"].strip(),
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason="stop",
        )
        usage = model_parse(CompletionUsage, completion["usage"])
        return ChatCompletion(
            id="chat" + completion["id"],
            choices=[choice],
            created=completion["created"],
            model=completion["model"],
            object="chat.completion",
            usage=usage,
        )
    
    def _create_chat_completion_stream(self, input_ids, gen_config) -> Iterator:
        completion = self.create_steam_completion(input_ids, gen_config)
        for i, output in enumerate(completion):
            _id, _created, _model = output["id"], output["created"], output["model"]
            if i == 0:
                choice = ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                )
                yield ChatCompletionChunk(
                    id=f"chat{_id}",
                    choices=[choice],
                    created=_created,
                    model=_model,
                    object="chat.completion.chunk",
                )

            if output["finish_reason"] is None:
                delta = ChoiceDelta(content=output["text"])
            else:
                delta = ChoiceDelta()

            choice = ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=output["finish_reason"],
            )
            yield ChatCompletionChunk(
                id=f"chat{_id}",
                choices=[choice],
                created=_created,
                model=_model,
                object="chat.completion.chunk",
            )
    
    def apply(self, messages: List[ChatCompletionMessageParam], **kwargs):
        max_tokens = kwargs["max_tokens"]
        messages = [_C.ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        input_ids = self.pipeline.tokenizer.encode_messages(messages, max_tokens)
        repetition_penalty=kwargs.get("frequency_penalty", 1)
        gen_config = _C.GenerationConfig(
            max_length=max_tokens,
            max_new_tokens=max_tokens - len(input_ids),
            max_context_length=max_tokens,
            do_sample=True,
            top_k=0,
            top_p=kwargs["top_p"] if kwargs["top_p"] is not None else 0.7,
            temperature=kwargs["temperature"] if kwargs["temperature"] is not None else 0.95,
            repetition_penalty=repetition_penalty if repetition_penalty > 0 else 1,
            num_threads=0,
        )
        return input_ids, gen_config

    def create_chat_completion(self, messages: List[ChatCompletionMessageParam], **kwargs) -> Union[Iterator, ChatCompletion]:
        input_ids, gen_config = self.apply(messages=messages, **kwargs)
        return (
            self._create_chat_completion_stream(input_ids, gen_config)
            if kwargs.get("stream", False)
            else self._create_chat_completion(input_ids, gen_config)
        )