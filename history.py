import streamlit as st
from pymongo import MongoClient
import pandas as pd
import os
import base64

mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

mongo_client = MongoClient(mongo_url)

# 使用数据库 rag
db = mongo_client["rag"]

# 使用集合 history
chats_collection = db["history"]

# 获取集合中的所有数据
all_data = chats_collection.find()

# 处理数据并转换为 DataFrame 格式
data_list = []
# 用于存储所有result_file的数据
result_file_list = []

for document in all_data:
    # 将数据结构转换为适合表格显示的格式
    doc = {
        "_id": str(document["_id"]),  # ObjectId 转换为字符串
        "start_time": document["start_time"].strftime("%Y-%m-%d %H:%M:%S"),  # 转换为字符串
        "end_time": document["end_time"].strftime("%Y-%m-%d %H:%M:%S"),  # 转换为字符串
        "source_specification": ", ".join(document["source_specification"]),  # 将列表转换为逗号分隔的字符串
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
    # 提取result_file
    result_file_list.append(document["result_file"])

# 转换为 pandas DataFrame
df = pd.DataFrame(data_list)


# 为result_file列生成下载按钮
def generate_download_button(result_file):
    # 如果文件存在，创建下载按钮
    if os.path.exists(result_file):
        return f'<a href="data:application/octet-stream;base64,{get_base64_of_file(result_file)}" download="{os.path.basename(result_file)}">Download</a>'
    else:
        return "File not found"

# 获取文件的 Base64 编码（为了使它能作为下载链接）
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 将 result_file 列转换为 HTML 下载按钮链接
df["Download_Link"] = df["result_file"].apply(generate_download_button)

# 转换 DataFrame 为 HTML 格式
html_table = df.to_html(escape=False, render_links=True)

# 添加滚动条到表格

st.write("### Result Data")
st.markdown(
    f"""
    <div style="overflow-x: auto; max-height: 400px;">
        {html_table}
    </div>
    """, 
    unsafe_allow_html=True
)