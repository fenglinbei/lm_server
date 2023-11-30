# LLM SERVER

当前分支使用的模型为
`chatglm3-6b-int8-ggml`

## 🐳 环境配置

### docker启动

在项目主目录下
`docker-compose up -d`
即可完成docker镜像的构建与启动



## 🤖 使用方式

### 参数配置

在项目根目录的`.env`修改以下参数

+ `PORT`: 服务器端口

+ `MODEL_NAME`: 若使用的是ggml模型，此处填入模型的`.bin`文件路径，若使用HF模型，此处填入包含`config.json`的模型文件夹

+ `MODEL_PATH`: 若使用的是ggml模型，此处填入原模型的包含`config.json`的模型文件夹，若使用HF模型，此处填入包含`config.json`的模型文件夹
