import chatglm_cpp

pipeline = chatglm_cpp.Pipeline("/workspace/checkpoints/chatglm3-6b/chatglm-ggml.bin", dtype="q4_0")
print(pipeline.chat(["你好,"]))
