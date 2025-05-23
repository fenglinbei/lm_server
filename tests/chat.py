import openai
from openai.openai_object import OpenAIObject

# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://192.168.1.40:7891/v1"

# List models API
models = openai.Model.list()
# print("Models:", models)

model = models["data"][0]["id"]

# Chat completion API
chat_completion: OpenAIObject = openai.ChatCompletion.create(
    # temperature=0.1,
    top_p=0.9,
    max_new_tokens=10,
    model=model,
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办"
        },
    ],
    # stop=["喝水"]
)

# print("Chat completion results:")
print(chat_completion["choices"][0]["message"]["content"])
print(chat_completion["usage"])

# chat_completion = openai.ChatCompletion.create(
#     top_p=0.9,
#     max_tokens=10,
#     model=model,
#     messages=[
#         {
#             "role": "user",
#             "content": "感冒了怎么办"
#         },
#     ],
#     stream=True,
#     # stop="水"
# )

# print("Chat completion streaming results:")
# # print(chat_completion)
# content = ""
# i = 0
# for c in chat_completion:
#     content = c.choices[0].delta.get("content", "")
#     print(content, flush=True, end="")
# print()
    # i += 1
    # if i == 10:
    #     break
