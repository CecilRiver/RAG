import streamlit as st
from pymongo import MongoClient
import pandas as pd
import os
import base64
from bson.objectid import ObjectId
import plotly.express as px
from datetime import datetime
import time

# Set page config
st.set_page_config(
    page_title="Result Manager",
    page_icon="📊",
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
    
    /* Button styling */
    .download-btn {
        display: inline-block;
        padding: 6px 12px;
        background-color: #59f240;
        color: white;
        border-radius: 5px;
        text-decoration: none;
        font-size: 0.8em;
        transition: all 0.3s;
    }
    
    .download-btn:hover {
        background-color: #25b30f;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
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
    
    /* Search box styling */
    .search-box {
        padding: 8px 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
        width: 100%;
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
</style>
""", unsafe_allow_html=True)

# Initialize session state for delete confirmation
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = None
if 'delete_success' not in st.session_state:
    st.session_state.delete_success = False
if 'delete_error' not in st.session_state:
    st.session_state.delete_error = False
if 'selected_row' not in st.session_state:
    st.session_state.selected_row = None

# MongoDB connection
mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"
mongo_client = MongoClient(mongo_url)
db = mongo_client["rag"]
chats_collection = db["history"]

# Function to load data from MongoDB
@st.cache_data(ttl=5)  # Cache for 5 seconds
def load_data():
    all_data = chats_collection.find()
    data_list = []
    
    for document in all_data:
        # Transform data for display
        doc = {
            "_id": str(document["_id"]),
            "start_time": document["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": document["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "source_specification": ", ".join(document["source_specification"]),
            "pieces": document["pieces"],
            "LLM_name": document["LLM_parameters"]["name"],
            "LLM_temperature": document["LLM_parameters"]["temperature"],
            "LLM_top_p": document["LLM_parameters"]["top_p"],
            "Knowledgebase_name": document["Knowledgebase_parameters"]["knowledgebase_name"],
            "Knowledgebase_semantic_based": document["Knowledgebase_parameters"]["semantic_based"],
            "Knowledgebase_keyword_based": document["Knowledgebase_parameters"]["keyword_based"],
            "Knowledgebase_embedding_model": document["Knowledgebase_parameters"]["embedding_model_name"],
            "result_file": document["result_file"]
        }
        data_list.append(doc)
    
    return pd.DataFrame(data_list)

# Function to delete a document by ID
def delete_document(doc_id):
    try:
        result = chats_collection.delete_one({"_id": ObjectId(doc_id)})
        if result.deleted_count > 0:
            st.session_state.delete_success = True
            return True
        else:
            st.session_state.delete_error = True
            return False
    except Exception as e:
        st.session_state.delete_error = True
        return False

# Function to generate styled download button
def generate_download_button(result_file):
    if os.path.exists(result_file):
        return f'<a href="data:application/octet-stream;base64,{get_base64_of_file(result_file)}" download="{os.path.basename(result_file)}" class="download-btn">Download</a>'
    else:
        return "<span style='color:#EF4444;'>File not found</span>"

# Get base64 of file
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Main app header with gradient background
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;padding:0;font-size:2.5rem;">📊 Result Manager</h1>
    <p style="opacity:0.8;margin-top:5px;">View, search, and manage result records from the database</p>
</div>
""", unsafe_allow_html=True)

# Load the data
df = load_data()

# Display success/error messages if they exist
if st.session_state.delete_success:
    st.markdown("""
    <div class="toast-success">
        <strong>✅ Success!</strong> Record deleted successfully.
    </div>
    """, unsafe_allow_html=True)
    # Reset after 3 seconds
    time.sleep(1)
    st.session_state.delete_success = False
    
if st.session_state.delete_error:
    st.markdown("""
    <div class="toast-error">
        <strong>❌ Error!</strong> Failed to delete the record.
    </div>
    """, unsafe_allow_html=True)
    # Reset after 3 seconds
    time.sleep(1)
    st.session_state.delete_error = False

# Display metrics
if not df.empty:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    
    # Metric 1: Total Records
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df)}</div>
        <div class="metric-label">Total Records</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric 2: Unique LLMs
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['LLM_name'].nunique()}</div>
        <div class="metric-label">Unique LLM Models</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric 3: Knowledge Bases
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Knowledgebase_name'].nunique()}</div>
        <div class="metric-label">Knowledge Bases</div>
    </div>
    """, unsafe_allow_html=True)
    
    # # Metric 4: Average Pieces
    # avg_pieces = round(df['pieces'].astype(int).mean(), 1)
    # st.markdown(f"""
    # <div class="metric-card">
    #     <div class="metric-value">{avg_pieces}</div>
    #     <div class="metric-label">Avg. Pieces</div>
    # </div>
    # """, unsafe_allow_html=True)
    
    # st.markdown('</div>', unsafe_allow_html=True)

# Create two columns for search and actions
col1, col2 = st.columns([2, 1])

# Search and Filter card
with col1:
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.subheader("🔍 Search and Filter")
    
    search_term = st.text_input("Search by keyword:", placeholder="Enter search term...")
    
    col_search1, col_search2 = st.columns([2, 1])
    with col_search1:
        search_field = st.selectbox(
            "Search in field:",
            ["All Fields", "_id", "LLM_name", "Knowledgebase_name", "start_time", "end_time", 
             "source_specification", "Knowledgebase_embedding_model"]
        )
    with col_search2:
        st.write("&nbsp;")  # Add some space
        refresh_button = st.button("🔄 Refresh Data", use_container_width=True)
        if refresh_button:
            st.cache_data.clear()
            st.session_state.confirm_delete = None
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Actions card
with col2:
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.subheader("⚙️ Actions")
    
    if not df.empty:
        selected_id = st.selectbox("Select a record:", df["_id"].tolist())
        
        # Show a view button and delete button
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            if st.button("📝 View Details", use_container_width=True):
                st.session_state.selected_row = selected_id
        
        with col_a2:
            if st.session_state.confirm_delete == selected_id:
                if st.button("❌ Confirm Delete", use_container_width=True, type="primary"):
                    if delete_document(selected_id):
                        st.session_state.confirm_delete = None
                        st.cache_data.clear()
                        st.rerun()
            else:
                if st.button("🗑️ Delete", use_container_width=True, type="secondary"):
                    st.session_state.confirm_delete = selected_id
                    st.rerun()
    else:
        st.info("No records available.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Apply search filters if provided
if search_term:
    if search_field == "All Fields":
        # Search in all string columns
        mask = pd.Series(False, index=df.index)
        for col in df.select_dtypes(include=['object']).columns:
            mask = mask | df[col].astype(str).str.contains(search_term, case=False, na=False)
        filtered_df = df[mask]
    else:
        # Search in specific column
        filtered_df = df[df[search_field].astype(str).str.contains(search_term, case=False, na=False)]
else:
    filtered_df = df

# Add download links to the DataFrame
filtered_df["Download"] = filtered_df["result_file"].apply(generate_download_button)

# Display record details if selected
if st.session_state.selected_row:
    selected_data = filtered_df[filtered_df["_id"] == st.session_state.selected_row].iloc[0]
    
    st.markdown('<div class="stcard">', unsafe_allow_html=True)
    st.subheader(f"📄 Record Details: {st.session_state.selected_row}")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.markdown("#### Session Information")
        st.markdown(f"**Start Time:** {selected_data['start_time']}")
        st.markdown(f"**End Time:** {selected_data['end_time']}")
        st.markdown(f"**Source:** {selected_data['source_specification']}")
        st.markdown(f"**Pieces:** {selected_data['pieces']}")
    
    with col_d2:
        st.markdown("#### Model Information")
        st.markdown(f"**LLM Name:** {selected_data['LLM_name']}")
        st.markdown(f"**Temperature:** {selected_data['LLM_temperature']}")
        st.markdown(f"**Top P:** {selected_data['LLM_top_p']}")
        st.markdown(f"**Knowledgebase:** {selected_data['Knowledgebase_name']}")
    
    st.markdown("#### Knowledge Base Configuration")
    st.markdown(f"**Semantic Based:** {selected_data['Knowledgebase_semantic_based']}")
    st.markdown(f"**Keyword Based:** {selected_data['Knowledgebase_keyword_based']}")
    st.markdown(f"**Embedding Model:** {selected_data['Knowledgebase_embedding_model']}")
    
    st.markdown("#### Result File")
    st.markdown(selected_data['Download'], unsafe_allow_html=True)
    
    if st.button("Close Details", use_container_width=True):
        st.session_state.selected_row = None
        st.rerun()
        
    st.markdown('</div>', unsafe_allow_html=True)

# Main Data Table
st.markdown('<div class="stcard">', unsafe_allow_html=True)
st.subheader(f"📋 Results ({len(filtered_df)} records)")

# Create a more compact display for the table
display_columns = [
    "_id", 
    "start_time", 
    "end_time", 
    "source_specification",
    "LLM_name", 
    "Knowledgebase_name",
    "Download"
]

# Convert DataFrame to HTML with custom styling
html_table = filtered_df[display_columns].to_html(escape=False, index=False)
html_table = html_table.replace('<table', '<table class="styled-table"')

# Display the styled table
st.markdown(
    f"""
    <div style="overflow-x: auto; max-height: 500px;">
        {html_table}
    </div>
    """, 
    unsafe_allow_html=True
)

# Show button to view all columns
if st.button("View All Columns", use_container_width=True):
    # All columns including download button
    display_all_columns = [
        "_id", 
        "start_time", 
        "end_time", 
        "source_specification",
        "pieces",
        "LLM_name", 
        "LLM_temperature", 
        "LLM_top_p",
        "Knowledgebase_name", 
        "Knowledgebase_semantic_based",
        "Knowledgebase_keyword_based",
        "Knowledgebase_embedding_model",
        "Download"
    ]
    
    # Create the full HTML table with all fields
    full_html_table = filtered_df[display_all_columns].to_html(escape=False, index=False)
    full_html_table = full_html_table.replace('<table', '<table class="styled-table"')
    
    st.markdown(
        f"""
        <div style="overflow-x: auto; max-height: 500px;">
            {full_html_table}
        </div>
        """, 
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #6B7280; font-size: 0.8em;">
    Result Manager © 2025 | Last updated: {}
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# import streamlit as st
# from pymongo import MongoClient
# import pandas as pd
# import os
# import base64

# mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# mongo_client = MongoClient(mongo_url)

# # 使用数据库 rag
# db = mongo_client["rag"]

# # 使用集合 history
# chats_collection = db["history"]

# # 获取集合中的所有数据
# all_data = chats_collection.find()

# # 处理数据并转换为 DataFrame 格式
# data_list = []
# # 用于存储所有result_file的数据
# result_file_list = []

# for document in all_data:
#     # 将数据结构转换为适合表格显示的格式
#     doc = {
#         "_id": str(document["_id"]),  # ObjectId 转换为字符串
#         "start_time": document["start_time"].strftime("%Y-%m-%d %H:%M:%S"),  # 转换为字符串
#         "end_time": document["end_time"].strftime("%Y-%m-%d %H:%M:%S"),  # 转换为字符串
#         "source_specification": ", ".join(document["source_specification"]),  # 将列表转换为逗号分隔的字符串
#         "pieces": document["pieces"],
#         "LLM_name": document["LLM_parameters"]["name"],
#         "LLM_temperature": document["LLM_parameters"]["temperature"],
#         "LLM_top_p": document["LLM_parameters"]["top_p"],
#         "Knowledgebase_name": document["Knowledgebase_parameters"]["knowledgebase_name"],
#         "Knowledgebase_semantic_based": document["Knowledgebase_parameters"]["semantic_based"],
#         "Knowledgebase_keyword_based": document["Knowledgebase_parameters"]["keyword_based"],
#         "Knowledgebase_embedding_model": document["Knowledgebase_parameters"]["embedding_model_name"],
#         "result_file": document["result_file"]
#     }
#     data_list.append(doc)
#     # 提取result_file
#     result_file_list.append(document["result_file"])

# # 转换为 pandas DataFrame
# df = pd.DataFrame(data_list)


# # 为result_file列生成下载按钮
# def generate_download_button(result_file):
#     # 如果文件存在，创建下载按钮
#     if os.path.exists(result_file):
#         return f'<a href="data:application/octet-stream;base64,{get_base64_of_file(result_file)}" download="{os.path.basename(result_file)}">Download</a>'
#     else:
#         return "File not found"

# # 获取文件的 Base64 编码（为了使它能作为下载链接）
# def get_base64_of_file(file_path):
#     with open(file_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")

# # 将 result_file 列转换为 HTML 下载按钮链接
# df["Download_Link"] = df["result_file"].apply(generate_download_button)

# # 转换 DataFrame 为 HTML 格式
# html_table = df.to_html(escape=False, render_links=True)

# # 添加滚动条到表格

# st.write("### Result Data")
# st.markdown(
#     f"""
#     <div style="overflow-x: auto; max-height: 400px;">
#         {html_table}
#     </div>
#     """, 
#     unsafe_allow_html=True
# )