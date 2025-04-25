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
import time
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Knowledge Base Manager",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f9f9fd;
        padding: 20px;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .stcard {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    
    /* Custom table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9em;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
    }
    
    .styled-table thead tr {
        background-color: #1E3A8A;
        color: white;
        text-align: left;
        position: sticky;
        top: 0;
    }
    
    .styled-table th,
    .styled-table td {
        padding: 12px 15px;
        border-bottom: 1px solid #dddddd;
    }
    
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f4f6;
    }
    
    .styled-table tbody tr:hover {
        background-color: #e5e7eb;
    }
    
    /* Custom metrics */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        flex: 1;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        color: #1E3A8A;
    }
    
    .metric-label {
        color: #6B7280;
        font-size: 0.9em;
    }
    
    /* Upload zone styling */
    .upload-zone {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
        background-color: #f9fafb;
        transition: all 0.3s;
    }
    
    .upload-zone:hover {
        border-color: #3B82F6;
        background-color: #f0f4ff;
    }
    
    /* Form styling */
    .form-group {
        margin-bottom: 15px;
    }
    
    .form-label {
        font-weight: 500;
        margin-bottom: 5px;
        display: block;
        color: #374151;
    }
    
    .form-hint {
        font-size: 0.8em;
        color: #6B7280;
        margin-top: 2px;
    }
    
    /* Toast notifications */
    .toast-success {
        padding: 10px 15px;
        background-color: #10B981;
        color: white;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .toast-error {
        padding: 10px 15px;
        background-color: #EF4444;
        color: white;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    /* File list styling */
    .file-list {
        margin: 10px 0;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        background-color: #f3f4f6;
        padding: 8px 12px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    
    .file-icon {
        margin-right: 10px;
        color: #3B82F6;
    }
    
    .file-name {
        flex: 1;
    }
    
    .file-size {
        color: #6B7280;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'create' not in st.session_state:
    st.session_state.create = False
if 'create_success' not in st.session_state:
    st.session_state.create_success = False
if 'selected_kb' not in st.session_state:
    st.session_state.selected_kb = None

# MongoDB connection
mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"
mongo_client = MongoClient(mongo_url)
db = mongo_client["rag"]
knowledgebase_collection = db["knowledgebase"]

# Main app header with gradient background
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;padding:0;font-size:2.5rem;">📚 Knowledge Base Manager</h1>
    <p style="opacity:0.8;margin-top:5px;">Create, view, and manage your knowledge bases</p>
</div>
""", unsafe_allow_html=True)

# Function to get knowledgebase data
@st.cache_data(ttl=5)  # Cache for 5 seconds
def get_kb_data():
    all_data = knowledgebase_collection.find()
    data_list = []
    for document in all_data:
        # Transform data for display
        doc = {
            "_id": str(document["_id"]),
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
    return pd.DataFrame(data_list)

# Function to save uploaded files
def save_uploaded_files(uploaded_files):
    if not uploaded_files:
        return [], []
    session_folder = f"knowledge/{uuid.uuid4()}"
    os.makedirs(session_folder, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return session_folder, file_paths

# Function for KB creation action
def on_create():
    st.session_state.create = True

# Load knowledge base data
df = get_kb_data()

# Display success message if needed
if st.session_state.create_success:
    st.markdown("""
    <div class="toast-success">
        <strong>✅ Success!</strong> Knowledge base created successfully.
    </div>
    """, unsafe_allow_html=True)
    # Reset after 3 seconds
    time.sleep(1)
    st.session_state.create_success = False

# Display metrics
if not df.empty:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    # Metric 1: Total Knowledge Bases
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df)}</div>
        <div class="metric-label">Knowledge Bases</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric 2: Average Chunk Size
    # avg_chunk = round(pd.to_numeric(df['chunk_size'], errors='coerce').mean(), 0)
    # st.markdown(f"""
    # <div class="metric-card">
    #     <div class="metric-value">{int(avg_chunk) if not pd.isna(avg_chunk) else 'N/A'}</div>
    #     <div class="metric-label">Avg. Chunk Size</div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Metric 3: Total Files
    try:
        total_files = df['source'].str.count(',').sum() + len(df)
    except:
        total_files = len(df)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_files}</div>
        <div class="metric-label">Total Files</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric 4: Latest Update
    latest_update = df['update_time'].max() if 'update_time' in df else 'N/A'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size:1.2em;">{latest_update.split(' ')[0] if latest_update != 'N/A' else 'N/A'}</div>
        <div class="metric-label">Latest Update</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Create two columns layout
col1, col2 = st.columns([3, 2])

# Existing Knowledge Bases section
with col1:
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.subheader("📋 Existing Knowledge Bases")
    
    if df.empty:
        st.info("No knowledge bases found. Create your first one using the form.")
    else:
        # Create the HTML table with styling
        display_columns = ["collection_name", "description", "embedding_model_name", "create_time"]
        html_table = df[display_columns].to_html(escape=False, index=False)
        html_table = html_table.replace('<table', '<table class="styled-table"')
        
        st.markdown(
            f"""
            <div style="overflow-x: auto; max-height: 300px;">
                {html_table}
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Add options to view details
        if not df.empty:
            st.markdown("#### Select Knowledge Base for Details")
            selected_name = st.selectbox("", df["collection_name"].tolist())
            if st.button("View Details", use_container_width=True):
                st.session_state.selected_kb = selected_name
        
    st.markdown('</div>', unsafe_allow_html=True)

    # Display KB details if selected
    if st.session_state.selected_kb:
        selected_data = df[df["collection_name"] == st.session_state.selected_kb].iloc[0]
        
        st.markdown('<div class="stcard">', unsafe_allow_html=True)
        st.subheader(f"📄 Knowledge Base Details: {st.session_state.selected_kb}")
        
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("#### Basic Information")
            st.markdown(f"**Name:** {selected_data['collection_name']}")
            st.markdown(f"**Description:** {selected_data['description']}")
            st.markdown(f"**Created:** {selected_data['create_time']}")
            st.markdown(f"**Last Updated:** {selected_data['update_time']}")
        
        with col_d2:
            st.markdown("#### Configuration")
            st.markdown(f"**Embedding Model:** {selected_data['embedding_model_name']}")
            st.markdown(f"**Chunk Size:** {selected_data['chunk_size']}")
            st.markdown(f"**Chunk Overlap:** {selected_data['chunk_overlap']}")
        
        st.markdown("#### Source Files")
        st.text(selected_data['source'])
        
        if st.button("Close Details", use_container_width=True):
            st.session_state.selected_kb = None
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

# Create New Knowledge Base Form
with col2:
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.subheader("🔨 Create New Knowledge Base")
    
    # Get existing collection names for validation
    collection_names = knowledgebase_collection.find({}, {"collection_name": 1, "_id": 0})
    collection_name_list = [doc["collection_name"] for doc in collection_names]
    
    # Form inputs with better styling
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label class="form-label">Collection Name</label>', unsafe_allow_html=True)
    collection_name = st.text_input("", placeholder="Enter collection name", key="collection_name_input", label_visibility="collapsed")
    st.markdown('<div class="form-hint">Choose a unique name for your knowledge base</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label class="form-label">Description</label>', unsafe_allow_html=True)
    description = st.text_area("", placeholder="Describe the purpose of this knowledge base", key="description_input", label_visibility="collapsed", height=80)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for chunk parameters
    chunk_col1, chunk_col2 = st.columns(2)
    
    with chunk_col1:
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        st.markdown('<label class="form-label">Chunk Size</label>', unsafe_allow_html=True)
        chunk_size = st.number_input("", min_value=500, max_value=2000, value=1000, key="chunk_size_input", label_visibility="collapsed")
        st.markdown('<div class="form-hint">Size of text chunks (500-2000)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chunk_col2:
        st.markdown('<div class="form-group">', unsafe_allow_html=True)
        st.markdown('<label class="form-label">Chunk Overlap</label>', unsafe_allow_html=True)
        chunk_overlap = st.number_input("", min_value=100, max_value=400, value=200, key="chunk_overlap_input", label_visibility="collapsed")
        st.markdown('<div class="form-hint">Overlap between chunks (100-400)</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # File Upload with better styling
    st.markdown('<label class="form-label">Upload Knowledge Files</label>', unsafe_allow_html=True)
    # st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", accept_multiple_files=True, label_visibility="collapsed")
    
    if not uploaded_files:
        st.markdown("""
            <div style="color: #6B7280;">
                <i class="fas fa-upload" style="font-size: 1.5em;"></i>
                <p>Drag and drop files here or click to browse</p>
                <p style="font-size: 0.8em;">Support for multiple file formats</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="file-list">', unsafe_allow_html=True)
        for file in uploaded_files:
            # Display file info with icons
            file_size = round(file.size / 1024, 1)
            file_ext = os.path.splitext(file.name)[1].lower()
            icon = "📄"
            if file_ext == ".pdf":
                icon = "📕"
            elif file_ext in [".doc", ".docx"]:
                icon = "📝"
            elif file_ext in [".xls", ".xlsx"]:
                icon = "📊"
            elif file_ext in [".txt", ".md"]:
                icon = "📃"
                
            st.markdown(f"""
                <div class="file-item">
                    <div class="file-icon">{icon}</div>
                    <div class="file-name">{file.name}</div>
                    <div class="file-size">{file_size} KB</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create Button
    if st.button("Create Knowledge Base", type="primary", use_container_width=True):
        on_create()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Processing logic when Create button is clicked
if st.session_state.create:
    with st.spinner("Creating knowledge base..."):
        # Validation
        validation_error = False
        
        if not collection_name:
            st.error("Collection Name cannot be empty")
            validation_error = True
        elif collection_name in collection_name_list:
            st.error("Collection Name already exists")
            validation_error = True
        elif not uploaded_files:
            st.warning("No files uploaded. Your knowledge base will be empty.")
            
        if not validation_error:
            # Text splitter
            splitter = TextSplitter(
                splitter_type = "recursive",
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
            )
            
            # Initialize Chroma vector database
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
                
                # Process files
                for file in st.session_state.knowledge_data_value:
                    if os.path.isfile(file):
                        docs = splitter.split_file_documents([file])
                    else:
                        st.error(f"Invalid file: {file}")
                        st.session_state.create = False
                        break
                    
                    # Add documents to vector store
                    vector_store_manager.add_documents(docs)
                
                # Create MongoDB entry
                data = {
                    "collection_name": collection_name,
                    "description": description,
                    "source": {
                        "file": st.session_state["knowledge_data_value"],
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                    "create_time": datetime.now(),
                    "update_time": datetime.now(),
                    "embedding_model_name": "nomic-embed-text"
                }
                
                knowledgebase_collection.insert_one(data)
                
                # Set success state and reset form
                st.session_state.create_success = True
                st.session_state.create = False
                
                # Force refresh to update the displayed data
                st.cache_data.clear()
                st.rerun()
            
    # Reset create state if we reach here without rerunning
    st.session_state.create = False

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #6B7280; font-size: 0.8em;">
    Knowledge Base Manager © 2025 | Last updated: {}
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
# import streamlit as st
# from src.rag.vector_store import VectorStoreManager
# from src.rag.text_splitter import TextSplitter
# import os
# from langchain.docstore.document import Document
# import uuid
# from PyPDF2 import PdfReader
# import pandas as pd


# from pymongo import MongoClient
# from datetime import datetime
# # 直接写死 MongoDB URI
# mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# # 连接到 MongoDB
# mongo_client = MongoClient(mongo_url)

# # 访问数据库和集合
# db = mongo_client["rag"]  # 使用数据库 rag
# knowledgebase_collection = db["knowledgebase"]  # 使用集合 knowledgebase

# # 获取集合中的所有数据
# all_data = knowledgebase_collection.find()

# # 处理数据并转换为 DataFrame 格式
# data_list = []
# for document in all_data:
#     # 将数据结构转换为适合表格显示的格式
#     doc = {
#         "_id": str(document["_id"]),  # ObjectId 转换为字符串
#         "collection_name": document["collection_name"],
#         "description": document["description"],
#         "source": document["source"] if isinstance(document["source"], str) else ", ".join(document["source"].get('file', [])),
#         "chunk_size": document["source"].get("chunk_size", "N/A") if isinstance(document["source"], dict) else "N/A",
#         "chunk_overlap": document["source"].get("chunk_overlap", "N/A") if isinstance(document["source"], dict) else "N/A",
#         "create_time": document["create_time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(document["create_time"], datetime) else "N/A",
#         "update_time": document["update_time"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(document["update_time"], datetime) else "N/A",
#         "embedding_model_name": document["embedding_model_name"]
#     }
#     data_list.append(doc)

# # 转换为 pandas DataFrame
# df = pd.DataFrame(data_list)

# # 显示 DataFrame
# with st.expander("KnowledgeBase Data"):
#     st.write("### KnowledgeBase Data")
#     st.dataframe(df)

# def save_uploaded_files(uploaded_files):
#     # 保存上传的文件并且返回它们的路径
#     if not uploaded_files:
#         return []
#     session_folder = f"knowledge/{uuid.uuid4()}"
#     os.makedirs(session_folder, exist_ok=True)
#     file_paths = []
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(session_folder, uploaded_file.name)
#         with open(file_path,"wb") as f:
#             f.write(uploaded_file.getbuffer())
#         file_paths.append(file_path)
#     return session_folder, file_paths



# st.write("### Create KnowledgeBase")


# # 每个文档至少 50-100 字，提高词频和信息量
# # 可设置参数
# # 知识库集合名称
# collection_name = st.text_input("Collection Name")
# # 文本分割大小 500-2000
# chunk_size = st.number_input("Chunk Size", min_value=500,max_value=2000,value=1000)
# # 文本重叠大小 100-400
# chunk_overlap = st.number_input("Chunk Overlap", min_value=100,max_value=400,value=200)
# # 知识库描述
# description = st.text_input("KnowledgeBase Description")

# # 上传文件(当前只对单个pdf格式文件进行了适配)
# uploaded_files = st.file_uploader("Upload Knowledge Files", accept_multiple_files=True)

# if 'create' not in st.session_state:
#     st.session_state.create = False
    
# def on_create():
#     st.session_state.create = True



# # 获取所有集合名称
# # 获取所有 collection_name 字段
# collection_names = knowledgebase_collection.find({}, {"collection_name": 1, "_id": 0})

# # 提取 collection_name 字段
# collection_name_list = [doc["collection_name"] for doc in collection_names]



# st.button("Create New KnowledgeBase", on_click = on_create)
# if st.session_state.create:
#     # 参数合法性检验
#     if not collection_name:
#         st.error("Collection Name can not be empty")
#         st.session_state.create = False
 
#     elif collection_name in collection_name_list:
#         st.error("Collection Name already exists")
#         st.session_state.create = False
#     else:    
#         #文本分割器
#         splitter = TextSplitter(
#             splitter_type = "recursive",
#             chunk_size = chunk_size,
#             chunk_overlap = chunk_overlap,
#         )


#         # 初始化Chroma向量数据库
#         vector_store_manager = VectorStoreManager(
#                     vector_store_type="chroma",
#                     collection_name=collection_name,
#                     embedding_model_name = "nomic-embed-text",
#                     embedding_type="llama"
#                 )

#         if uploaded_files:
#             session_folder, data_value = save_uploaded_files(uploaded_files)
#             st.session_state["knowledge_session_folder"] = session_folder
#             st.session_state["knowledge_data_value"] = data_value

#             # 文件分割
#             for file in st.session_state.knowledge_data_value:
#                 if(os.path.isfile(file)):
#                     docs = splitter.split_file_documents([file])
#                 else:
#                     raise ValueError("Invalid data_value for 'file' data_task. Must be a file path or list of file paths.")     
#                 # 分割后文本加入知识库中
#                 vector_store_manager.add_documents(docs)

#         data ={
#         "collection_name":collection_name,
#         "description": description,
#         "source":{
#             "file":st.session_state["knowledge_data_value"],
#             "chunk_size":chunk_size,
#             "chunk_overlap":chunk_overlap,
#         } ,
#         "create_time":datetime.now(),
#         "update_time":datetime.now(),
#         "embedding_model_name":"nomic-embed-text"
#         }

#         knowledgebase_collection.insert_one(data)

#         st.success(f"KnowledgeBase {collection_name} has been created.")
#         st.session_state.create = False





                    




# # 知识库清空
# vector_store_manager.clear_vector_store()







# # 知识文件地址
# # data = r"C:\Users\ROOT\Desktop\连铸\连铸设备主要技术参数.pdf"
# data = r"C:\Users\ROOT\Desktop\连铸\6机6流连铸工程连铸机技术附件要点.pdf"






# # 检索示例
# retriever = vector_store_manager.as_retriever()

# print(retriever.get_relevant_documents("推理步骤"))

