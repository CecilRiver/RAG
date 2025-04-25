# RAG 文档知识抽取系统

基于 Streamlit 的文档知识抽取与 RAG 问答系统，从操作手册、规范文档中抽取设备信息、业务拓扑和控制逻辑。

## 功能

- 上传 PDF / Word 文档并构建知识库
- Chroma 向量检索与 LightRAG 知识图谱
- 支持 OpenAI、DeepSeek、HuggingFace、Ollama 等模型
- 设备信息、业务拓扑、控制逻辑三类抽取结果展示
- MongoDB 保存对话历史与知识库信息
- 提供消融实验脚本（`scripts/ablation_study.py`）
- Streamlit Web 界面，支持 Docker 部署

## 技术栈

- Python 3.11 / Streamlit
- LangChain、ChromaDB、LightRAG
- MongoDB
- PyPDF2、docx2txt
- OpenAI / DeepSeek / HuggingFace / Ollama

## 目录结构

```
app.py                        Streamlit 主界面
streamlit_app.py              带登录页的示例界面
history.py / knowledge.py     历史记录与知识库页面
src/llm/                      各模型封装与 LLM 管线
src/rag/                      文档加载、切分、向量库、爬虫
src/run_rag_pipeline.py       RAG 流程编排
src/run_lightrag_pipeline.py  LightRAG 流程编排
scripts/                      示例、测试与消融实验脚本
config.yaml                   管线配置
.env.example                  环境变量示例
Dockerfile / render.yaml      部署配置
```

## 本地运行

环境要求：Python 3.11、MongoDB。

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

按需填写 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY`、`HUGGINGFACE_API_KEY`、`PINECONE_API_KEY`、`MONGO_*` 等配置。

3. 启动 MongoDB，默认连接 `mongodb://localhost:27017/rag`。

4. 启动应用

```bash
streamlit run app.py
```

默认访问地址为 `http://localhost:8501`。

## Docker

```bash
docker build -t rag-app .
docker run --env-file .env -p 8501:8501 rag-app
```

## 说明

- `uploads/`、`knowledge/`、`chroma_db/`、`lightrag_data/`、`result/`、`references/` 等数据目录不随仓库提供，运行时生成。
- 请勿把 `.env` 或 API Key 提交到仓库。
- `streamlit_app.py` 中的默认登录账号仅用于本地演示。
