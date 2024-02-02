import numpy as np

from typing import Optional, List
from tritonclient import utils as client_utils
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer
import concurrent.futures

WEIGHT2NPDTYPE = {
    "fp32": np.float32,
    "fp16": np.float16,
}


class EmbeddingClient:
    DEFAULT_MAX_RESP_WAIT_S = 120
    embed_version = "local_v0.0.1_20230525_6d4019f1559aef84abc2ab8257e1ad4c"

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_version: str,
        tokenizer_path: str,
        resp_wait_s: Optional[float] = None,
    ):
        self._server_url = server_url
        self._model_name = model_name
        self._model_version = model_version
        self._response_wait_t = self.DEFAULT_MAX_RESP_WAIT_S if resp_wait_s is None else resp_wait_s
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_embedding(self, sentences, max_length=512):
        # Setting up client
    
        inputs_data = self._tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='np')
        inputs_data = {k: v for k, v in inputs_data.items()}
    
        client = InferenceServerClient(url=self._server_url)
        model_config = client.get_model_config(self._model_name, self._model_version)
        model_metadata = client.get_model_metadata(self._model_name, self._model_version)
    
        inputs_info = {tm.name: tm for tm in model_metadata.inputs}
        outputs_info = {tm.name: tm for tm in model_metadata.outputs}
        output_names = list(outputs_info)
        outputs_req = [InferRequestedOutput(name_) for name_ in outputs_info]
        infer_inputs = []
        for name_ in inputs_info:
            data = inputs_data[name_]
            infer_input = InferInput(name_, data.shape, inputs_info[name_].datatype)
    
            target_np_dtype = client_utils.triton_to_np_dtype(inputs_info[name_].datatype)
            data = data.astype(target_np_dtype)
    
            infer_input.set_data_from_numpy(data)
            infer_inputs.append(infer_input)
    
        results = client.infer(
            model_name=self._model_name,
            model_version=self._model_version,
            inputs=infer_inputs,
            outputs=outputs_req,
            client_timeout=120,
        )
        y_pred = {name_: results.as_numpy(name_) for name_ in output_names}
        embeddings = y_pred["output"][:,0]
        norm_arr = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norm_arr
        return embeddings_normalized.tolist()
    
    def getModelVersion(self):
        return self.embed_version


class YouDaoLocalEmbeddings:
    def __init__(self, server_url, model_name, tokenizer_path, max_length=512, batch_size=16):
        self.embedding_client = EmbeddingClient(
            server_url=server_url,
            model_name=model_name,
            model_version='1',
            resp_wait_s=120,
            tokenizer_path=tokenizer_path
            )
        self.max_length = max_length
        self.batch_size = batch_size

    def _get_embedding(self, queries):
        embeddings = self.embedding_client.get_embedding(queries, max_length=self.max_length)
        return embeddings

    def _get_len_safe_embeddings(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                future = executor.submit(self._get_embedding, batch)
                futures.append(future)
            for future in futures:
                embeddings = future.result()
                all_embeddings += embeddings
        return all_embeddings

    @property
    def embed_version(self):
        return self.embedding_client.getModelVersion()
