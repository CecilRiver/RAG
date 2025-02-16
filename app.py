import streamlit as st
from pymongo import MongoClient
from src.run_rag_pipeline import RunChatbot
import os
import json
import uuid
from dotenv import load_dotenv
import json
import uuid
import logging
import time 
from io import StringIO
import shutil  # 导入用于文件夹清理的模块
import requests
from streamlit_lottie import st_lottie

# 从 .env 文件加载环境变量
load_dotenv()

# 使用环境变量构建 MongoDB URI
username = os.getenv("MONGO_USERNAME") 
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER") 
database = os.getenv("MONGO_DB") 
app_name = os.getenv("MONGO_APP_NAME") 
    
# GitHub API 密钥
github_username = os.getenv('GITHUB_USERNAME')
github_personal_token = os.getenv('GITHUB_PERSONAL_TOKEN')

# AI 模型 API 密钥
openai_api_key = os.getenv('OPENAI_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
huggingface_api_key_2 = os.getenv('HUGGINGFACE_API_KEY_2')


os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# 向量数据库 API 密钥
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

# 研究论文 API 密钥
elsevier_api_key = os.getenv('ELSEVIER_API_KEY')
ieee_api_key =  os.getenv('IEEE_API_KEY')
elsevier_api_secret = os.getenv('ELSEVIER_API_SECRET')

# # MongoDB API keys
# mongo_username = os.getenv('MONGO_USERNAME')
# mongo_password = os.getenv('MONGO_PASSWORD')
# mongo_cluster = os.getenv('MONGO_CLUSTER')
# mongo_db = os.getenv('MONGO_DB')
# mongo_collections = os.getenv('MONGO_COLLECTIONS')
# mongo_app_name = os.getenv('MONGO_APP_NAME')

# if not all([username, password, cluster, database, app_name]):
#     raise ValueError("One or more MongoDB environment variables are missing.")

# MongoDB Connection URI
# mongo_uri = f"mongodb://{username}:{password}@{cluster}/{database}?retryWrites=true&w=majority&appName={app_name}"

# # Connect to MongoDB
# mongo_client = MongoClient(mongo_uri)

# # Access Database and Collections
# db = mongo_client[database]
# chats_collection = db["chats"]  # Replace "chats" with the name of your collection


# 直接写死 MongoDB URI
mongo_uri = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# 连接到 MongoDB
mongo_client = MongoClient(mongo_uri)

# 访问数据库和集合
db = mongo_client["rag"]  # 使用数据库 rag
chats_collection = db["history"]  # 使用集合 history

# Streamlit 应用标题
st.title("RAG For Extraction")

# 初始化聊天机器人
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# CSS 样式
def apply_styles():
    st.markdown("""
    <style>
        /* General Body Styling */
        body {
            background-color: #f2f2f2; /* Light grey background */
            font-family: 'Arial', sans-serif;
            color: black; /* Standard black text for contrast */
        }

        /* Sidebar Customization */
        section[data-testid="stSidebar"] {
            background-color: #e6e6e6; /* Light grey sidebar */
            color: black; /* Black text for readability */
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #333; /* Dark grey for headings in the sidebar */
        }

        /* Input and Button Styles */
        .stTextInput, .stTextArea, .stSelectbox, .stRadio {
            background-color: #ffffff !important; /* White background for inputs */
            border: 1px solid #ccc !important; /* Subtle border */
            border-radius: 5px !important;
            padding: 10px !important;
            color: black; /* Black text */
        }

        .stButton button {
            background-color: #4caf50 !important; /* Subtle green for buttons */
            color: white !important;
            border: none;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
        }

        .stButton button:hover {
            background-color: #45a049 !important; /* Slightly darker green on hover */
        }

        /* Header Styling */
        h1, h2, h3, h4 {
            color: #333; /* Dark grey for headers */
            font-weight: 600;
        }

        /* Chat Bubble Styles */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 10px;
        }

        .chat-bubble {
            max-width: 80%;
            padding: 12px 18px;
            border-radius: 10px;
            margin-bottom: 10px;
        }

        .user-message {
            background-color: #d9fdd3; /* Light green for user messages */
            color: black;
            align-self: flex-end;
        }

        .assistant-message {
            background-color: #e6e6e6; /* Light grey for assistant messages */
            color: black;
            align-self: flex-start;
        }

        /* Footer Styling */
        footer {
            text-align: center;
            font-size: 12px;
            color: #666;
            margin-top: 50px;
        }
    </style>
    """, unsafe_allow_html=True)

# 加载 Lottie 动画
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 应用样式
apply_styles()

# 添加 Lottie 动画
def add_header_animation():
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_ksu5dpjr.json"  # Clean chatbot animation
    animation_data = load_lottie_url(lottie_url)
    if animation_data:
        st_lottie(animation_data, height=200, key="header_animation")

add_header_animation()


# 带交互选项的侧边栏
st.sidebar.markdown("""
<div style="
    background-color: #585858; 
    padding: 10px; 
    border-radius: 8px; 
    text-align: center; 
    margin-bottom: 15px;
    border: 1px solid #dcdcdc;">
    <h2 style="color: white; font-family: 'Arial', sans-serif; margin: 0;">
        <strong>CustomGPT</strong>! 🚀
    </h2>
</div>
""", unsafe_allow_html=True)


    
# 日志记录器设置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 辅助函数
def initialize_chatbot(
    data_task=None,
    vector_store_type="chroma",
    model_type="openai",
    data_value=None,
    openai_model=None,
    llama_model_path=None,
    huggingface_model=None,
    deepseek_model=None,
    embedding_type="llama",
    rag_type = "LightRAG",  # 新参数，用于在不同框架之间切换
    ollama_embedding_model = None
):

    if model_type is None:
        raise ValueError("Model type is required but not provided.")
    if model_type == "openai" and openai_model is None:
        raise ValueError("OpenAI model is required but not provided.")
        # 如果没有提供 data_task 或 data_value：
        #     默认使用 "LangChain" 作为 RAG 类型
    
    if rag_type == "LightRAG":
        # 使用基于 LightRAG 的框架
        from src.run_lightrag_pipeline import RunLightRAGChatbot  # 假设为 LightRAG 创建了一个单独的类

        st.session_state.chatbot = RunLightRAGChatbot(
            model_type=model_category.lower(),  # 使用提供的模型类别
            working_dir="lightrag_data",  # LightRAG 工作文件的目录
            openai_model=openai_model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            huggingface_model_name=huggingface_model,
            huggingface_tokenizer_name=huggingface_model, # 默认为模型名称
            ollama_model_name=llama_model_path,  # 假设 Ollama 模型具有类似的命名结构
            ollama_host="http://localhost:11434",  # 示例 Ollama 主机；根据需要调整
            ollama_embedding_model=ollama_embedding_model,  # 使用提供的嵌入类型
            data_task=data_task,
            data_value=data_value,

            github_access_token=os.getenv("GITHUB_PERSONAL_TOKEN"),
            ieee_api_key=os.getenv("IEEE_API_KEY"),
            elsevier_api_key = os.getenv('ELSEVIER_API_KEY'),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY'),
        )

        # 设置 LightRAG 组件
        st.session_state.chatbot.setup_data()
        st.session_state.chatbot.setup_lightrag()
    elif rag_type == "LangChain":
        
        # 使用初始的基于 RunChatbot 的框架
        from src.run_rag_pipeline import RunChatbot
                
       # """初始化或重新初始化聊天机器人实例。"""
        st.session_state.chatbot = RunChatbot(
            model_type=model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            use_rag=bool(data_task),
            data_task=data_task,
            data_value=data_value,
            vector_store_type=vector_store_type,
            embedding_type=embedding_type,
            temperature=0.7,
            github_access_token=os.getenv("GITHUB_PERSONAL_TOKEN"),
            ieee_api_key=os.getenv("IEEE_API_KEY"),
            elsevier_api_key = os.getenv('ELSEVIER_API_KEY'),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY'),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model = deepseek_model,
            model=openai_model,
            model_path=llama_model_path,
            model_name=huggingface_model,
        )
        st.session_state.chatbot.setup_data()
        st.session_state.chatbot.setup_vector_store()
        st.session_state.chatbot.setup_llm_pipeline()


    elif rag_type == "None" or data_task is None or data_value is None:
        
        # 使用初始的基于 RunChatbot 的框架
        from src.run_rag_pipeline import RunChatbot
                
        # """初始化或重新初始化聊天机器人实例。"""
        st.session_state.chatbot = RunChatbot(
            model_type=model_type,
            api_key=os.getenv("OPENAI_API_KEY"),
            use_rag=bool(data_task),
            data_task=data_task,
            data_value=data_value,
            vector_store_type=vector_store_type,
            embedding_type=embedding_type,
            temperature=0.7,
            github_access_token=os.getenv("GITHUB_PERSONAL_TOKEN"),
            ieee_api_key=os.getenv("IEEE_API_KEY"),
            elsevier_api_key = os.getenv('ELSEVIER_API_KEY'),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY'),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_model = deepseek_model,
            model=openai_model,
            model_path=llama_model_path,
            model_name=huggingface_model,
        )
        st.session_state.chatbot.setup_llm_pipeline()
        
        
def save_uploaded_files(uploaded_files):
    """Save uploaded files and return their paths."""
    if not uploaded_files:
        return []
    session_folder = f"uploads/{uuid.uuid4()}"
    os.makedirs(session_folder, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return session_folder, file_paths


# Sidebar for RAG Input Options
st.sidebar.header("Additional Input Options")
data_icon = st.sidebar.radio(
    "Select Input Type",
    ["None", "Upload Document", "Web Link", "GitHub Repository", "Research Papers Topic", "Solve GitHub Issues"],
    help="Choose the type of additional input to enhance the chatbot's knowledge base.",
)

uploaded_files = None
data_value = None
data_task = None
rag_ready = False

if data_icon == "Upload Document":
    uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)
    if uploaded_files:
        data_task = "file"
        session_folder, data_value = save_uploaded_files(uploaded_files)
        st.session_state["session_folder"] = session_folder  # Store the folder path for cleanup
        
elif data_icon == "Web Link":
    web_link = st.sidebar.text_input("Enter Web Link (e.g., 'https://example.com')")
    if web_link.startswith("http://") or web_link.startswith("https://"):
        data_task = "url"
        data_value = web_link
        
elif data_icon == "GitHub Repository":
    github_repo = st.sidebar.text_input("Enter GitHub Repo URL (e.g., 'https://github.com/user/repo.git')")
    if github_repo.startswith("http://") or github_repo.startswith("https://"):
        data_task = "github_repo"
        data_value = github_repo
        
elif data_icon == "Research Papers Topic":
    research_topic = st.sidebar.text_input("Enter Research Paper Topic")
    if research_topic.strip():
        data_task = "research_papers"
        data_value = research_topic
        
elif data_icon == "Solve GitHub Issues":
    github_issues_repo = st.sidebar.text_input("Enter GitHub Repo for Issues (e.g., 'user/repo')")
    if github_issues_repo.strip():
        data_task = "github_issues"
        data_value = github_issues_repo

    
# 侧边栏：机器学习模型
st.sidebar.header("Large Language Models")
model_category = st.sidebar.radio(
    "Select a Model Category:",
    ["OpenAI", "HuggingFace", "Ollama", "DeepSeek"],
    help="Choose the category of machine learning model you'd like to use.",
)

openai_model = None
huggingface_model_path = None
ollama_model_path = None
model_ready = False  # 标志位，用于确认模型准备就绪
rag_type = None
deepseek_model = None

# 模型选择
if model_category == "OpenAI":
    openai_model = st.sidebar.selectbox(
        "Choose an OpenAI Model:",
        ["Select a Model", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "o1", "o1-mini", "gpt-3.5-turbo"],
        help="Select an OpenAI model to use."
    )
    if openai_model == "Select a Model": # 确保用户选择有效的模型
        openai_model = None
        st.sidebar.warning("Please select a valid OpenAI model.")
    else:
        st.sidebar.success(f"Selected OpenAI Model: `{openai_model}`")
        model_ready = True

elif model_category == "DeepSeek":
    deepseek_model = st.sidebar.text_input(
        "Enter DeepSeek Model:",
        placeholder="e.g., deepseek-chat",
        help="Select a DeepSeek model to use"
    )
    if deepseek_model.strip() in ["deepseek-chat", "deepseek-coder"]:  # 确保用户已提供输入
        model_ready = True
        st.sidebar.success(f"Selected DeepSeek Model: `{deepseek_model}`")
    else:
        deepseek_model = None
        st.sidebar.warning("Please provide a valid DeepSeek model.")
        
elif model_category == "HuggingFace":
    huggingface_model_path = st.sidebar.text_input(
        "Enter HuggingFace Model Path:",
        placeholder="e.g., meta-llama/Llama-2-7b-hf",
        help="Provide the path to a HuggingFace model from the model hub."
    )
    if huggingface_model_path.strip(): # 确保用户已提供输入
        model_ready = True
        st.sidebar.success(f"Selected HuggingFace Model Path: `{huggingface_model_path}`")
    else:
        huggingface_model_path = None
        st.sidebar.warning("Please provide a valid HuggingFace model path.")

elif model_category == "Ollama":
    ollama_model_path = st.sidebar.text_input(
        "Enter Ollama Model Path:",
        placeholder="e.g., ollama/gpt-j",
        help="Provide the path to an Ollama model."
    )
    if ollama_model_path.strip():  # 确保用户已提供输入
        model_ready = True
        st.sidebar.success(f"Selected Ollama Model Path: `{ollama_model_path}`")
    else:
        ollama_model_path = None
        st.sidebar.warning("Please provide a valid Ollama model path.")
        

# 执行 RAG 或初始化模型按钮
if (data_task and data_value) or model_ready:  # 确保 RAG 或模型输入已准备好
    rag_type = st.sidebar.selectbox(
        "Choose RAG Framework:",
        ["None", "LangChain", "LightRAG"],
        help="Select the framework for Retrieval-Augmented Generation."
    )
    # LightRAG 模式的子类别
    light_rag_mode = None
    if rag_type == "LightRAG":
        with st.sidebar.expander("Configure LightRAG Mode", expanded=False):
            st.markdown(
                "<small style='color: gray;'>Select a mode for LightRAG query execution:</small>",
                unsafe_allow_html=True
            )
            light_rag_mode = st.selectbox(
                "LightRAG Mode:",
                ["naive", "local", "global", "hybrid", "mix"],
                help=(
                    "Modes available for LightRAG:\n"
                    "- naive: Basic retrieval\n"
                    "- local: Local search\n"
                    "- global: Global search\n"
                    "- hybrid: Combines local and global search\n"
                    "- mix: Combines knowledge graph and vector search"
                ),
            )
    rag_ready = st.sidebar.button("Perform RAG or Initialize Model")

# print(f"Selected OpenAI model: {openai_model}")
# print(f"Data task: {data_task}, Data value: {data_value}")
# print(f"RAG type selected: {rag_type}")

# 仅在确认后初始化聊天机器人
if rag_ready:  # 在继续之前检查两个标志位
    try:
        if model_category.lower() == "openai" and not openai_model:
            st.error("Please select a valid OpenAI model.")
        elif model_category.lower() == "deepseek" and not deepseek_model:
            st.error("Please provide a valid DeepSeek model.")
        elif model_category.lower() == "huggingface" and not huggingface_model_path:
            st.error("Please provide a valid HuggingFace model path.")
        elif model_category.lower() == "ollama" and not ollama_model_path:
            st.error("Please provide a valid Ollama model path.")
        else:
            initialize_chatbot(
                data_task=data_task,
                data_value=data_value,
                model_type=model_category.lower(),
                openai_model=openai_model,
                llama_model_path=ollama_model_path,
                huggingface_model=huggingface_model_path,
                deepseek_model = deepseek_model,
                rag_type=rag_type,
            )
            st.sidebar.success(f"Chatbot initialized with model `{model_category}` and RAG task `{data_task}`, using '{rag_type}.")
    finally:
        # 清理上传的文件和文件夹
        session_folder = st.session_state.get("session_folder")
        if session_folder and os.path.exists(session_folder):
            shutil.rmtree(session_folder)  # 删除文件夹及其内容
            st.session_state.pop("session_folder", None)  # 清除会话状态
            st.sidebar.info("Temporary files have been cleaned up.")

# 显示消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if user_prompt := st.chat_input("Your prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_response = ""

        try:
            if not st.session_state.chatbot:
                # initialize_chatbot()
                initialize_chatbot(
                    data_task=data_task,
                    data_value=data_value,
                    model_type=model_category.lower(),
                    openai_model=openai_model,
                    llama_model_path=ollama_model_path,
                    huggingface_model=huggingface_model_path,
                    deepseek_model = deepseek_model,
                    rag_type=rag_type,
                )
            if rag_type == "LightRAG":
                answer = st.session_state.chatbot.chat(f"Question: {user_prompt}", mode = light_rag_mode)
                sources = []  # LightRAG 没有来源
            else:
                result = st.session_state.chatbot.chat(f"Question: {user_prompt}", with_sources=True)

                answer = result.get("result", "Sorry, I couldn't process that.")
                sources = result.get("sources", [])

                if isinstance(answer, str):
                    answer = answer
                elif hasattr(answer, 'content'):
                    answer = answer.content
                    # sources = result.additional_kwargs.get('sources', [])
                else:
                    answer = answer['result'].content
                 
        
            # 显示助手的回复
            for token in answer:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)


            ## Display sources if available
            # if sources:
            #     sources_placeholder.markdown("**Sources:**")
            #     unique_sources = list(set(sources))  # Remove duplicates by converting to a set and back 
            #     source_links = []
            #     for idx, source in enumerate(unique_sources, start=1):
            #         # Check if the source is a valid URL and display it as a clickable link
            #         if isinstance(source, str):
            #             if source.startswith("http"):
            #                 source_links.append(f"{idx}. [{source}]({source})")
            #             else:
            #                 source_links.append(f"{idx}. {source}")
            #         else:
            #             # Handle non-string sources (e.g., dicts or objects)
            #             source_links.append(f"{idx}. {str(source)}")
            #     # Combine all source links and display them
            #     sources_placeholder.markdown("\n".join(source_links))



            # 显示并格式化来源
            formatted_sources = ""
            if sources:
                formatted_sources = "\n\n**Sources:**\n"
                unique_sources = list(set(sources))
                for idx, source in enumerate(unique_sources, start=1):
                    if isinstance(source, str):
                        if source.startswith("http"):
                            formatted_sources += f"{idx}. [{source}]({source})\n"
                        else:
                            formatted_sources += f"{idx}. {source}\n"
                    else:
                        formatted_sources += f"{idx}. {str(source)}\n"
                sources_placeholder.markdown(formatted_sources)
                
                                            
            # Save the response to session and MongoDB
            # st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # 将完整的回复（包括来源）保存到会话状态
            st.session_state.messages.append({
                "role": "assistant", 
                # "content": full_response + formatted_sources if formatted_sources else full_response,
                "content": full_response,
                "sources": sources  # Store sources separately for future reference
            })
            chats_collection.insert_one(
                {
                    "question": user_prompt,
                    "answer": full_response,
                    "sources": sources,
                    "data_task": data_task,
                    "data_value": data_value,
                }
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error: {e}")

# 清除聊天历史
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    chats_collection.delete_many({})
    st.sidebar.success("Chat history cleared.")
