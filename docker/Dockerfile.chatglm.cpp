FROM nvcr.io/nvidia/pytorch:23.08-py3

WORKDIR /workspace/
ENV PYTHONPATH /workspace/
COPY requirements.txt /workspace/
COPY . /workspace

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /workspace/requirements.txt -i https://mirror.baidu.com/pypi/simple
RUN pip install bitsandbytes --upgrade -i https://mirror.baidu.com/pypi/simple

RUN cd /workspace/chatglm.cpp && CMAKE_ARGS="-DGGML_CUBLAS=ON" pip install .
