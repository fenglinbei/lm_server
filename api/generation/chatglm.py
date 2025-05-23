import gc
import json
import re
from typing import List, Union

import torch
from loguru import logger
from transformers.generation.logits_process import LogitsProcessor

from api.generation.utils import apply_stopping_strings
from api.utils.protocol import Role, ChatMessage

try:
    from chatglm_cpp import  _C
except:
    pass

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response


def process_response_v3(output: str, use_tool: bool = False) -> Union[str, dict]:
    content = ""
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            content = content.replace("[[训练时间]]", "2023年")
        else:
            if use_tool:
                content = "\n".join(content.split("\n")[1:-1])

                def tool_call(**kwargs):
                    return kwargs

                parameters = eval(content)
                content = {
                    "name": metadata.strip(),
                    "arguments": json.dumps(parameters, ensure_ascii=False)
                }
            else:
                content = {
                    "name": metadata.strip(),
                    "content": content
                }
    return content


def check_is_chatglm(model) -> bool:
    return "GLMBlock" in getattr(model, "_no_split_modules", [])


@torch.inference_mode()
def generate_stream_chatglm(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):
    prompt = params["prompt"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    gen_kwargs = {
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids
        else:
            output_ids = total_ids[input_echo_len:]
        response = tokenizer.decode(output_ids)
        response = process_response(response)

        yield {
            "text": response,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": total_len - input_echo_len,
                "total_tokens": total_len,
            },
            "finish_reason": None,
        }

    # TODO: ChatGLM stop when it reaches max length
    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_stream_chatglm_v3(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):  
    prompt: List[ChatMessage] = params["prompt"]
    functions = params["functions"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    messages = process_chatglm_messages(prompt, functions=functions)
    if functions:
        logger.debug(f"==== Messages with tools ====\n{messages}")

    query, role = messages[-1]["content"], messages[-1]["role"]

    inputs = tokenizer.build_chat_input(query, history=messages[:-1], role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len >= model.config.seq_length:
        logger.warning(f"Input length larger than {model.config.seq_length}")

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
    ]

    gen_kwargs = {
        "max_length": min(max_new_tokens + input_echo_len, model.config.seq_length),
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    generator = model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs)
    for total_ids in generator:
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            response, stop_found = apply_stopping_strings(response, ["<|observation|>"])

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": "function_call" if stop_found else None,
            }

            if stop_found:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()
    
def generate_stream_chatglm_v3_cpp(
    model,
    tokenizer,
    params,
    device,
    context_len=2048,
    stream_interval=2,
):  
    prompt: List[ChatMessage] = params["prompt"]
    functions = params["functions"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    messages = process_chatglm_messages(prompt, functions=functions)
    if functions:
        logger.debug(f"==== Messages with tools ====\n{messages}")

    query, role = messages[-1]["content"], messages[-1]["role"]

    inputs = tokenizer.build_chat_input(query, history=messages[:-1], role=role)
    input_ids = inputs["input_ids"].tolist()[0]
    input_echo_len = len(inputs["input_ids"][0])

    gen_config = _C.GenerationConfig(
            max_length=max_new_tokens,
            max_context_length=input_echo_len,
            do_sample=True if temperature > 1e-5 else False,
            top_k=0,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_threads=0,
        )
    
    n_past = 0
    output_ids = []
    response = ""
    response_len = 0
    # print(input_ids)
    # print(len(input_ids), gen_config.max_length)
    while len(input_ids) < gen_config.max_length:
        # print(n_past, input_echo_len)
        next_token_id = model.model.generate_next_token(input_ids, gen_config, n_past, input_echo_len)
        n_past = len(input_ids)
        input_ids.append(next_token_id)
        
        output_ids.append(next_token_id)
        response_len = len(output_ids)

        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            response, stop_found = apply_stopping_strings(response, ["<|observation|>"])

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": response_len,
                    "total_tokens": response_len + input_echo_len,
                },
                "finish_reason": "function_call" if stop_found else None,
            }

            if stop_found:
                break
        
        if next_token_id == model.model.config.eos_token_id:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": response_len,
            "total_tokens": response_len + input_echo_len,
        },
        "finish_reason": "stop",
    }
    yield ret


def process_chatglm_messages(messages: List[ChatMessage], functions: Union[dict, List[dict]] = None) -> List[dict]:
    _messages = messages
    messages = []

    if functions:
        functions = list(map(lambda function: function.dict() if not isinstance(function, dict) else function, functions))
        messages.append(
            {
                "role": Role.SYSTEM,
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": functions
            }
        )

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        if role == Role.FUNCTION:
            messages.append({"role": "observation", "content": content})

        elif role == Role.ASSISTANT and func_call is not None:
            for response in content.split("<|assistant|>"):
                metadata, sub_content = response.split("\n", maxsplit=1)
                messages.append({"role": role, "metadata": metadata, "content": sub_content.strip()})
        else:
            messages.append({"role": role, "content": content})
    return messages
