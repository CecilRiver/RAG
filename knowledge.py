import streamlit as st
from src.rag.vector_store import VectorStoreManager
from src.rag.text_splitter import TextSplitter
import os
from langchain.docstore.document import Document
import uuid
from PyPDF2 import PdfReader
import pandas as pd


from pymongo import MongoClient
from datetime import datetime
# 直接写死 MongoDB URI
mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# 连接到 MongoDB
mongo_client = MongoClient(mongo_url)

# 访问数据库和集合
db = mongo_client["rag"]  # 使用数据库 rag
knowledgebase_collection = db["knowledgebase"]  # 使用集合 knowledgebase

# 获取集合中的所有数据
all_data = knowledgebase_collection.find()

# 处理数据并转换为 DataFrame 格式
data_list = []
for document in all_data:
    # 将数据结构转换为适合表格显示的格式
    doc = {
        "_id": str(document["_id"]),  # ObjectId 转换为字符串
        "collection_name": document["collection_name"],
        "description": document["description"],
        "source": document["source"] if isinstance(document["source"], str) else ", ".join(document["source"].get('file', [])),
        "chunk_size": document["source"].get("chunk_size", "N/A") if isinstance(document["source"], dict) else "N/A",
        "chunk_overlap": document["source"].get("chunk_overlap", "N/A") if isinstance(document["source"], dict) else "N/A",
        "create_time": document["create_time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(document["create_time"], datetime) else "N/A",
        "update_time": document["update_time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(document["update_time"], datetime) else "N/A",
        "embedding_model_name": document["embedding_model_name"]
    }
    data_list.append(doc)

# 转换为 pandas DataFrame
df = pd.DataFrame(data_list)

# 显示 DataFrame
with st.expander("KnowledgeBase Data"):
    st.write("### KnowledgeBase Data")
    st.dataframe(df)

def save_uploaded_files(uploaded_files):
    # 保存上传的文件并且返回它们的路径
    if not uploaded_files:
        return []
    session_folder = f"knowledge/{uuid.uuid4()}"
    os.makedirs(session_folder, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return session_folder, file_paths



st.write("### Create KnowledgeBase")


# 每个文档至少 50-100 字，提高词频和信息量
# 可设置参数
# 知识库集合名称
collection_name = st.text_input("Collection Name")
# 文本分割大小 500-2000
chunk_size = st.number_input("Chunk Size", min_value=500,max_value=2000,value=1000)
# 文本重叠大小 100-400
chunk_overlap = st.number_input("Chunk Overlap", min_value=100,max_value=400,value=200)
# 知识库描述
description = st.text_input("KnowledgeBase Description")

# 上传文件(当前只对单个pdf格式文件进行了适配)
uploaded_files = st.file_uploader("Upload Knowledge Files", accept_multiple_files=True)

if 'create' not in st.session_state:
    st.session_state.create = False
    
def on_create():
    st.session_state.create = True



# 获取所有集合名称
# 获取所有 collection_name 字段
collection_names = knowledgebase_collection.find({}, {"collection_name": 1, "_id": 0})

# 提取 collection_name 字段
collection_name_list = [doc["collection_name"] for doc in collection_names]



st.button("Create New KnowledgeBase", on_click = on_create)
if st.session_state.create:
    # 参数合法性检验
    if not collection_name:
        st.error("Collection Name can not be empty")
        st.session_state.create = False
 
    elif collection_name in collection_name_list:
        st.error("Collection Name already exists")
        st.session_state.create = False
    else:    
        #文本分割器
        splitter = TextSplitter(
            splitter_type = "recursive",
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
        )


        # 初始化Chroma向量数据库
        vector_store_manager = VectorStoreManager(
                    vector_store_type="chroma",
                    collection_name=collection_name,
                    embedding_model_name = "nomic-embed-text",
                    embedding_type="llama"
                )

        if uploaded_files:
            session_folder, data_value = save_uploaded_files(uploaded_files)
            st.session_state["knowledge_session_folder"] = session_folder
            st.session_state["knowledge_data_value"] = data_value

            # 文件分割
            for file in st.session_state.knowledge_data_value:
                if(os.path.isfile(file)):
                    docs = splitter.split_file_documents([file])
                else:
                    raise ValueError("Invalid data_value for 'file' data_task. Must be a file path or list of file paths.")     
                # 分割后文本加入知识库中
                vector_store_manager.add_documents(docs)

        data ={
        "collection_name":collection_name,
        "description": description,
        "source":{
            "file":st.session_state["knowledge_data_value"],
            "chunk_size":chunk_size,
            "chunk_overlap":chunk_overlap,
        } ,
        "create_time":datetime.now(),
        "update_time":datetime.now(),
        "embedding_model_name":"nomic-embed-text"
        }

        knowledgebase_collection.insert_one(data)

        st.success(f"KnowledgeBase {collection_name} has been created.")
        st.session_state.create = False





                    




# # 知识库清空
# vector_store_manager.clear_vector_store()







# # 知识文件地址
# # data = r"C:\Users\ROOT\Desktop\连铸\连铸设备主要技术参数.pdf"
# data = r"C:\Users\ROOT\Desktop\连铸\6机6流连铸工程连铸机技术附件要点.pdf"






# # 检索示例
# retriever = vector_store_manager.as_retriever()

# print(retriever.get_relevant_documents("推理步骤"))

