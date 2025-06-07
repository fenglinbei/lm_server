from functools import partial
from typing import Iterator

import anyio
import asyncio
from inspect import isgenerator, isasyncgen, iscoroutine
from fastapi import APIRouter, Depends, Request, HTTPException
from loguru import logger
from sse_starlette import EventSourceResponse

from api.models import GENERATE_ENGINE
from api.utils.compat import model_dump
from api.utils.protocol import ChatCompletionCreateParams, Role
from api.utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher,
    create_error_response
)

chat_router = APIRouter(prefix="/chat")


def get_engine():
    yield GENERATE_ENGINE


@chat_router.post("/completions", dependencies=[Depends(check_api_key)])
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine=Depends(get_engine),
):
    """Creates a completion for the chat message"""
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")

    request = await handle_request(request, engine.stop)
    request.max_tokens = request.max_tokens or 1024

    params = model_dump(request, exclude={"messages"})
    params |= dict(
        prompt_or_messages=request.messages,
        echo=False,
    )
    logger.debug(f"==== request ====\n{params}")

    iterator_or_completion = engine.create_chat_completion(params)

    logger.debug(type(iterator_or_completion))

    try:
        if iscoroutine(iterator_or_completion):
            completion = await iterator_or_completion  # 执行协程
            import time
            while iscoroutine(completion):
                completion = await completion
                logger.debug(completion)
                time.sleep(1)
            return completion
        elif isasyncgen(iterator_or_completion):
            async def async_iterator():
                async for chunk in iterator_or_completion:
                    yield chunk
            iterator = async_iterator()
        elif isgenerator(iterator_or_completion):
        # 场景3：同步生成器，转换为异步迭代
            async def sync_iterator_wrapper():
                loop = asyncio.get_event_loop()
                while True:
                    try:
                        chunk = await loop.run_in_executor(None, next, iterator_or_completion)
                        yield chunk
                    except StopIteration:
                        break
            iterator = sync_iterator_wrapper()
        else:
            return iterator_or_completion
        
        send_chan, recv_chan = anyio.create_memory_object_stream(10)

        # 返回 EventSourceResponse
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator,  # 使用动态生成的迭代器
            ),
        )
    
    except Exception as err:
        logger.error(err)
        logger.exception(err)
        return create_error_response(500, "Internal Server Error")
