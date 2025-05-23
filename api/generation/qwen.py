import json
import re
from copy import deepcopy
from typing import List, Union

from fastapi import HTTPException
from loguru import logger
from transformers import PreTrainedTokenizer

from api.generation.utils import parse_messages
from api.utils.protocol import Role, ChatMessage

TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}"""

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def build_qwen_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[ChatMessage],
    context_len: int = 8192,
    max_new_tokens: int = 256,
    functions: List[dict] = None,
) -> List[int]:
    """ https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py """
    query, history = process_qwen_messages(messages, functions)
    if query is _TEXT_COMPLETION_CMD:
        return build_last_message_input(tokenizer, history)
    else:
        for q, r in history:
            messages.extend([ChatMessage(role=Role.USER, content=q), ChatMessage(role=Role.ASSISTANT, content=r)])
        messages.append(ChatMessage(role=Role.USER, content=query))

    max_input_tokens = context_len - max_new_tokens
    system, rounds = parse_messages(messages)
    system = "You are a helpful assistant." + system  # fix system prompt

    im_start_tokens, im_end_tokens = [tokenizer.im_start_id], [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content):
        return tokenizer.encode(
            role, allowed_special=set()
        ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_tokens_part = _tokenize_str("system", system)
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    for r in rounds[::-1]:
        round_tokens = []
        for message in r:
            if round_tokens:
                round_tokens += nl_tokens

            if message.role == Role.USER:
                content_tokens = im_start_tokens + _tokenize_str("user", message.content) + im_end_tokens
            else:
                content_tokens = im_start_tokens + _tokenize_str("assistant", message.content) + im_end_tokens

            round_tokens.extend(content_tokens)

        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            if history_tokens:
                history_tokens = nl_tokens + history_tokens

            history_tokens = round_tokens + history_tokens  # concat left
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + nl_tokens + history_tokens
    if messages[-1].role != Role.ASSISTANT:
        input_tokens += nl_tokens + im_start_tokens + tokenizer.encode("assistant") + nl_tokens
    return input_tokens[-max_input_tokens:]  # truncate left


def check_is_qwen(model) -> bool:
    return "QWenBlock" in getattr(model, "_no_split_modules", [])


def process_qwen_messages(messages: List[ChatMessage], functions: Union[dict, List[dict]] = None):
    if all(m.role != Role.USER for m in messages):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: Expecting at least one user message.",
        )

    messages = deepcopy(messages)
    default_system = "You are a helpful assistant."
    system = ""
    if messages[0].role == Role.SYSTEM:
        system = messages.pop(0).content.lstrip("\n").rstrip()
        if system == default_system:
            system = ""

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get("name", "")
            name_m = func_info.get("name_for_model", name)
            name_h = func_info.get("name_for_human", name)
            desc = func_info.get("description", "")
            desc_m = func_info.get("description_for_model", desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info["parameters"], ensure_ascii=False),
            )

            tools_text.append(tool)
            tools_name_text.append(name_m)

        tools_text = "\n\n".join(tools_text)
        tools_name_text = ", ".join(tools_name_text)
        system += "\n\n" + REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        )
        system = system.lstrip("\n").rstrip()

    dummy_thought = {
        "en": "\nThought: I now know the final answer.\nFinal answer: ",
        "zh": "\nThought: 我会作答了。\nFinal answer: ",
    }

    _messages = messages
    messages = []
    for m_idx, m in enumerate(_messages):
        role, content, func_call = m.role, m.content, m.function_call
        if content:
            content = content.lstrip("\n").rstrip()
        if role == Role.FUNCTION:
            if (len(messages) == 0) or (messages[-1].role != Role.ASSISTANT):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role assistant before role function.",
                )
            messages[-1].content += f"\nObservation: {content}"
            if m_idx == len(_messages) - 1:
                messages[-1].content += "\nThought:"
        elif role == Role.ASSISTANT:
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid request: Expecting role user before role assistant.",
                )
            last_msg = messages[-1].content
            last_msg_has_zh = len(re.findall(r"[\u4e00-\u9fff]+", last_msg)) > 0

            if func_call is None:
                if functions:
                    content = dummy_thought["zh" if last_msg_has_zh else "en"] + content
            else:
                f_name, f_args = func_call.name, func_call.arguments
                if not content:
                    if last_msg_has_zh:
                        content = f"Thought: 我可以使用 {f_name} API。"
                    else:
                        content = f"Thought: I can use {f_name}."

            if messages[-1].role == Role.USER:
                messages.append(
                    ChatMessage(role=Role.ASSISTANT, content=content.lstrip("\n").rstrip())
                )
            else:
                messages[-1].content += content
        elif role == Role.USER:
            messages.append(
                ChatMessage(role=Role.USER, content=content.lstrip("\n").rstrip())
            )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid request: Incorrect role {role}."
            )

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == Role.USER:
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail="Invalid request")

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == Role.USER and messages[i + 1].role == Role.ASSISTANT:
            usr_msg = messages[i].content.lstrip("\n").rstrip()
            bot_msg = messages[i + 1].content.lstrip("\n").rstrip()
            if system and (i == len(messages) - 2):
                usr_msg = f"{system}\n\nQuestion: {usr_msg}"
                system = ""
            for t in dummy_thought.values():
                t = t.lstrip("\n")
                if bot_msg.startswith(t) and ("\nAction: " in bot_msg):
                    bot_msg = bot_msg[len(t):]
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid request: Expecting exactly one user (or function) role before every assistant role.",
            )
    if system:
        assert query is not _TEXT_COMPLETION_CMD
        query = f"{system}\n\nQuestion: {query}"
    return query, history


def parse_response(response):
    func_name, func_args = "", ""
    i = response.rfind("\nAction:")
    j = response.rfind("\nAction Input:")
    k = response.rfind("\nObservation:")

    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + "\nObservation:"  # Add it back.
        k = response.rfind("\nObservation:")
        func_name = response[i + len("\nAction:"): j].strip()
        func_args = response[j + len("\nAction Input:"): k].strip()

    if func_name:
        function_call = {"name": func_name, "arguments": func_args}
        return response[:k], function_call

    z = response.rfind("\nFinal Answer: ")
    if z >= 0:
        response = response[z + len("\nFinal Answer: "):]
    return response, None


def build_last_message_input(tokenizer: PreTrainedTokenizer, history: list):
    im_start = "<|im_start|>"
    im_end = "<|im_end|>"
    prompt = f"{im_start}system\nYou are a helpful assistant.{im_end}"
    for i, (query, response) in enumerate(history):
        query = query.lstrip("\n").rstrip()
        response = response.lstrip("\n").rstrip()
        prompt += f"\n{im_start}user\n{query}{im_end}"
        prompt += f"\n{im_start}assistant\n{response}{im_end}"
    prompt = prompt[:-len(im_end)]
    logger.debug(f"==== Prompt with tools ====\n{prompt}")
    return tokenizer.encode(prompt)
