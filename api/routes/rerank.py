import numpy as np
from loguru import logger
from fastapi import APIRouter, Depends, HTTPException
from openai.types.create_embedding_response import Usage

from api.config import SETTINGS
from api.core.rerank import RerankerModel
from api.models import RERANK_MODEL
from api.utils.protocol import CreateRerankerParams, RerankResult, CreateRerankerResponse
from api.utils.request import check_api_key

from utils.compat import model_dump

rerank_router = APIRouter()


def get_rerank_engine():
    yield RERANK_MODEL


@rerank_router.post("/rerank", dependencies=[Depends(check_api_key)])
async def create_reranks(
    request: CreateRerankerParams,
    engine: RerankerModel = Depends(get_rerank_engine),
):

    logger.info(f"Get rerank request: {str(model_dump(request))}")

    query = request.query
    passages = request.passages

    query_inputs = engine.tokenizer.encode_plus(query, truncation=False, padding=False)
    query_len = len(query_inputs['input_ids'])
    
    if query_len > 400:
        raise HTTPException(status_code=400, detail=f"Query {query_len} is too long . Please make sure your query less than 400 tokens!")
    
    rerank_idx, scores, passage_tokens = engine.rerank(query=query, query_inputs=query_inputs, passages=passages)
    

    rerank_result = RerankResult(
        rerank_idx=rerank_idx,
        scores=scores,
        object="rerank"
    )

    return CreateRerankerResponse(
        data=rerank_result,
        model=SETTINGS.reranker_name,
        object="list",
        usage=Usage(prompt_tokens=query_len, total_tokens=query_len + passage_tokens),
    )
