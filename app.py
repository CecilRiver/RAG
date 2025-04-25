import streamlit as st
from pymongo import MongoClient
from src.run_rag_pipeline import RunChatbot
import os
import json
import uuid
from dotenv import load_dotenv
import logging
import time
from io import StringIO
import shutil
import requests
from streamlit_lottie import st_lottie
from src.llm.llm_pipeline import LLMPipeline
from PyPDF2 import PdfReader
from src.rag.vector_store import VectorStoreManager
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import jieba
import math
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Interaction Logic Extraction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
        padding: 0;
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    .sidebar-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 15px 10px;
        color: white;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #3B82F6;
    }
    
    /* Upload zone styling */
    .upload-zone {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 25px;
        text-align: center;
        margin: 20px 0;
        background-color: #f9fafb;
        transition: all 0.3s;
    }
    
    .upload-zone:hover {
        border-color: #3B82F6;
        background-color: #f0f4ff;
    }
    
    /* Section headings */
    .section-heading {
        font-size: 1.2em;
        color: #1E3A8A;
        margin-bottom: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Result panels */
    .result-panel {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
    }
    
    .panel-title {
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 8px;
    }
    
    /* Status indicators */
    .status-ready {
        display: inline-block;
        background-color: #10B981;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        margin-top: 5px;
    }
    
    .status-processing {
        display: inline-block;
        background-color: #F59E0B;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8em;
        margin-top: 5px;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #334155;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8em;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Button styling */
    .primary-button {
        background-color: #3B82F6;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 500;
        text-align: center;
        display: block;
        width: 100%;
    }
    
    .primary-button:hover {
        background-color: #1E3A8A;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
    
    /* Chat message styling */
    .chat-message {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }

    /* Parameter label */
    .parameter-label {
        font-weight: 500;
        color: #334155;
        margin-bottom: 5px;
    }
    
    .parameter-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #3B82F6;
        text-align: center;
        margin: 5px 0;
    }
    
    /* Success message */
    .success-message {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# MongoDB connection
mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"
mongo_client = MongoClient(mongo_url)
db = mongo_client["rag"]  
chats_collection = db["history"]

# Main app header with gradient background
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;padding:0;font-size:2.5rem;">⚙️ Interaction Logic Extraction System</h1>
    <p style="opacity:0.8;margin-top:5px;">Extract device information, business topology, and control logic from operational specifications</p>
</div>
""", unsafe_allow_html=True)

# Sidebar styling
st.sidebar.markdown("""
<div class="sidebar-header">
    <h3 style="margin:0;padding:0;">Configuration Panel</h3>
    <p style="margin-top:5px;font-size:0.9em;opacity:0.8;">Customize extraction parameters</p>
</div>
""", unsafe_allow_html=True)

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define prompts
chat_prompt_1 = PromptTemplate(
    input_variables = ["context","question"],
    template = """
            
            这是**工控系统文本**，**所有分析都基于此文本**：{question}
            这是检索到的相关信息，**如果存在工控系统文本中不存在的内容请忽略**：{context}

            ## 角色简介:
            - 角色: 工控系统单点设备与属性提取大师
            - 语言: 中文
            - 描述:专注于工控系统中单点设备的识别与属性提取，精准分析各设备的名称、定位与类型，确保完整理解系统中的设备结构，为下一步的业务拓扑的提取做准备。

            ## 核心目标:
            - 第一步，明确工控系统文本中单点设备识别的目标和范围。你需要根据目标工控系统的特点，分析系统中涉及控制、监测和执行的单点设备，如控制器、操作员站、工程师站、被控物理过程、传感器和执行器等，为后续属性提取奠定基础。
            - 第二步，分析各设备的设备名称，此名称是唯一的。
            - 第三步，分析设备在该段文本中的设备定位，定位是不唯一的，可能是行为主体也可能是行为客体。
            - 第四步，分析设备的设备类型，可能是控制器、操作员站、工程师站、被控物理过程、传感器或执行器，设备类型是唯一的。
            - 第五步，结合前四步的分析，系统化提取设备的核心属性，并以清晰的结构进行整理，确保数据的可用性与可操作性。

            ## 约束条件:
            - 推理过程必须逻辑严密。每一步分析应具备层层递进的逻辑关系，确保设备属性提取的准确性与完整性。
            - 设备识别需精确。避免误分类或遗漏，确保所有核心设备都被识别并提取。
            - 过程模型分析需深入，明确设备的设备名称、设备定位及设备类型。
            - 避免遗漏细节，确保数据的完整性和可操作性。

            ## 重要补充：
            - **如果检索到的相关信息（{context}）与任务无关或无助于提取设备属性，可以忽略该信息，仅基于工控系统文本（{question}）进行分析**。
            - 如果某些设备属性无法从工控系统文本中提取出来，则在输出格式中填充“不存在”。不要编造信息。
            - 如果工控系统文本中不存在任何可提取的单点设备，则输出“【不存在单点设备】”。
            - 如果工控系统文本中存在可提取的单点设备，则必须完整提取每一个设备，并按照规定的格式输出，不得遗漏。

            ## 专项技能:
            - 精准识别单点设备: 能够从复杂的工控系统文本中精准识别控制器、操作员站、工程师站、被控物理过程、传感器和执行器等设备，并准确归类。
            - 系统化逻辑提取能力: 能够从多个角度分析设备的属性，并以清晰的结构整理输出，确保数据的高效利用。
            - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别设备及其属性，确保最终提取的逻辑严谨且无遗漏。

            ## 处理流程:
            - 作为工控系统单点设备与属性提取大师，你将**以提供的工控系统文本为主体**，并结合检索到的相关信息作为辅助，运用你的“专项技能”能力分析设备名称、设备定位和设备类型。
            - 第一步，你需要一步步思考并推理，分析提供的工控系统文本和检索到的相关信息，分析检索到的相关信息中是否存在不属于提供的工控系统文本中的内容，如果存在，忽略检索到的相关信息中的内容。
                例如：火焰切割机在检索到的相关信息中出现过，但提供的工控系统文本中没有提及过火焰收割机，则忽略火焰收割机的相关内容
            - 第二步，你需要一步步思考并推理，从**工控系统文本中**识别出所有的单点设备。
            - 第三步，你需要一步步思考并推理，详细分析各个单点设备的设备名称，此名称是唯一的，确保设备输入输出的完整描述。
            - 第四步，你需要一步步思考并推理，详细分析各个单点设备的设备类型，此类型是唯一的，可能是控制器、操作员站、工程师站、被控物理过程、传感器或执行器。
            - 第五步，你需要一步步思考并推理，结合以上四步你的推理过程，综合整理设备属性信息，确保每一个提取出的设备都能在提供的工控系统文本中找到，同时系统化提取并确保数据完整，并且将提取出的单点设备按输出格式表述出来。

            ## 输出格式:
            【推理过程】<该设备的提取过程：200字>
            【设备名称】<提取出的设备名称>
            【设备定位】<行为主体/行为客体/行为主体和行为客体>
            【设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>


    """
)


chat_prompt_2 = PromptTemplate(
    input_variables = ["device","text"],
    template = """

        这是**工控系统文本**，**所有分析都基于此文本**：{text}
        这是已经提取出的单点设备：{device}

        ## 角色简介:
        - 角色: 工控系统业务拓扑提取大师
        - 语言: 中文
        - 描述: 专注于工控系统中的业务拓扑结构提取，精准识别系统中行为主体和行为客体之间发生的行为内容，并细致分析行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值，确保全面理解系统的控制与反馈机制，为后续行为上下文的提取提供关键数据支持。
       
        ## 核心目标:
        - 第一步，明确业务拓扑分析的目标和范围。你需要从工控系统文本中识别各类控制行为，包括控制指令、数据读取/写入、反馈机制等，为后续分析奠定基础，同时结合已经提取出的单点设备，明确每个控制行为的行为主体和行为客体。
        - 第二步，深入分析各类行为的行为类型，行为类型为读或者写，行为类型是唯一的。
        - 第三步，分析行为的行为目标类型，行为目标类型为改变行为客体的输入、改变行为客体的输出或改变行为客体的设定值，行为目标类型是唯一的。
        - 第四步，分析行为的行为目标参数名称，行为目标参数名称可以是开关状态、温度设定值、输入电压或者其他类似目标参数名称，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
        - 第五步，分析行为的行为目标参数值，行为目标参数值可以是开、关、调高、调低或者其他类似目标参数值，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
        - 第六步，结合前五步的分析，系统化提取业务拓扑数据，并整理输出，以支持系统优化与控制策略制定。

        ## 约束条件:
        - 推理过程必须逻辑严密。每一步分析应具备层层递进的逻辑关系，确保业务拓扑提取的准确性与完整性。
        - 业务拓扑识别需精确。避免误分类或遗漏，确保所有业务拓扑都被识别并提取。
        - 推理分析需深入，明确行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值。
        - 行为目标参数名称不可以缺少，如果提取不出来，可以将其值设置为null。
        - 行为目标参数值不可以缺少，如果提取不出来，可以将其值设置为null。
        - 避免遗漏细节，确保数据的完整性和可操作性。

        ## 重要补充：
        - 如果某些业务拓扑的相关属性数据无法从工控系统文本中提取出来，则在输出格式中填充“不存在”。不要编造信息。
        - 如果工控系统文本中不存在任何可提取的业务拓扑结构，则输出“【不存在业务拓扑】”。
        - 如果工控系统文本中存在可提取的业务拓扑结构，则必须完整提取每一个业务拓扑结构，并按照规定的格式输出，不得遗漏。

        ## 专项技能:
        - 精准识别业务行为: 能够从复杂的工控系统文本中精准提取业务拓扑，确保分析全面。
        - 系统化拓扑提取能力: 能够从多个角度分析行为的相互关系，并以清晰的结构整理输出，确保数据的高效利用。
        - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别业务行为及其属性，确保最终提取的逻辑严谨且无遗漏。

        ## 处理流程:
        - 作为工控系统业务拓扑提取大师，你将从提供的工控系统文本中，结合已经提取出的单点设备，运用你的“专项技能”能力精准识别系统中行为主体和行为客体之间发生的行为内容，并细致分析行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值
        - 第一步，你需要一步步思考并推理，从文本中识别业务行为，并明确它们的作用，同时结合已经提取出的单点设备，明确每个业务行为的行为主体和行为客体。
        - 第二步，你需要一步步思考并推理，各类行为的行为类型，行为类型为读或者写，行为类型是唯一的，确保行为的精准描述。
        - 第三步，你需要一步步思考并推理，分析行为的行为目标类型，行为目标类型为改变行为客体的输入、改变行为客体的输出或改变行为客体的设定值，行为目标类型是唯一的。
        - 第四步，你需要一步步思考并推理，分析行为的行为目标参数名称，行为目标参数名称可以是开关状态、温度设定值、输入电压或这其他类似目标参数名称，具体要看文本中的行为是什么样的如果提取不出来，则设置为null。
        - 第五步，你需要一步步思考并推理，分析行为的行为目标参数值，行为目标参数值可以是开、关、调高、调低或者其他类似目标参数值，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
        - 第六步，你需要一步步思考并推理，结合前五步的分析，系统化提取业务拓扑数据，并整理输出，以支持系统优化与控制策略制定，并且要把提取出的业务拓扑按照输出格式进行输出。

    
        ## 输出格式:
    
        【行为主体的设备名称】<设备名称>
        【行为主体的设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>

        【行为客体的设备名称】<设备名称>
        【行为客体的设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>

        【行为类型】<读/写>
        【行为内容】<提取出的行为内容>
        【行为目标类型】<改变行为客体的输入/改变行为客体的输出/改变行为客体的设定值>
        【行为目标参数名称】<开关状态/温度设定值/输入电压/其他类似目标参数名称.../null>
        【行为目标参数值】<开/关/调高/调低/其他类似目标参数值.../null>


    """
)

chat_prompt_3 = PromptTemplate(
    input_variables = ["text","action"],
    template = """

    这是**工控系统文本**，**所有分析都基于此文本**：{text}
    这是已经提取出的业务拓扑：{action}

    ## 角色简介:
    - 角色: 工控系统业务逻辑提取大师
    - 语言: 中文
    - 描述: 专注于工控系统中的业务（行为）逻辑提取，精准分析行为的触发原因、触发类型及触发结果，确保完整理解系统的控制流程和反馈机制，为优化系统控制策略提供关键数据支持。

    ## 核心目标:
    - 第一步，明确业务逻辑分析的目标和范围。你需要结合工控系统文本和已经提取出的业务拓扑，识别每个业务拓扑中业务行为的触发原因、触发类型及触发结果。 
    - 第二步，识别行为的触发原因，是什么导致了该行为的发生。
    - 第二步，识别行为的触发类型，明确系统中的业务逻辑是周期式（定期执行的控制行为）还是中断式（由事件或特定条件触发的行为），触发类型唯一。
    - 第三步，识别行为的触发结果，该行为导致了什么结果。
    - 第五步，结合前三步的分析，系统化提取业务逻辑，并整理输出，以支持系统优化与控制策略制定。

    ## 约束条件:
    - 推理过程必须逻辑严密。确保对每个行为的原因和结果分析层层递进，保证业务逻辑的完整性和准确性。
    - 触发类型识别需精准。确保正确区分周期式和中断式触发方式，避免误分类。
    - 确保业务逻辑的可操作性。所有提取的信息需具有实际指导意义，确保可用于优化控制策略。
    - 如果按照输出格式表述的时候，有表格对应参数不存在或者提取不来，用null进行占位，不要空着



    ## 专项技能:
    - 精准识别行为的触发类型: 能够从复杂的工控系统文本中精准提取周期式触发和中断式触发的业务逻辑，确保逻辑分析全面准确。
    - 系统化逻辑提取能力: 能够从多个角度分析行为的上下文，并以清晰的结构整理输出，确保数据的高效利用。
    - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别业务逻辑及其触发机制，确保最终提取的逻辑严谨且无遗漏。

    ## 处理流程:
    - 作为工控系统业务逻辑提取大师，你将从提供的工控系统文本中，结合已经提取出的业务拓扑，运用你的“专项技能”能力分析行为上下文的触发原因、触发类型及触发结果。
    - 第一步，你需要一步步思考并推理，从文本中识别行为的触发原因，是什么导致了该行为的发生。
    - 第二步，你需要一步步思考并推理，详细识别行为的触发类型，明确系统中的业务逻辑是周期式（定期执行的控制行为）还是中断式（由事件或特定条件触发的行为），触发类型是唯一的。
    - 第三步，你需要一步步思考并推理，识别行为的触发结果，该行为导致了什么结果。
    - 第四步，你需要一步步思考并推理，结合以上三步你的推理过程，综合整理业务逻辑数据，系统化提取并确保数据完整，以便后续优化系统控制逻辑，并且将提取出的业务逻辑按输出格式表述出来。
    请注意：**如果按照输出格式表述的时候，有表格对应参数不存在或者提取不来，用null进行占位，不要空着**

    

    ## 输出格式:
| **行为主体的设备名称** | **行为主体的设备类型** | **行为客体的设备名称** | **行为客体的设备类型** | **行为类型** | **行为内容** | **行为目标类型** | **行为目标参数名称** | **行为目标参数值** | **行为触发原因** | **行为触发类型** | **行为触发结果** |
|-----------------------|------------------------|------------------------|------------------------|--------------|--------------|------------------|----------------------|---------------------|-------------------|-------------------|-------------------|
| <设备名称>            | <控制器/操作员站/工程师站/被控物理过程/传感器/执行器> | <设备名称>            | <控制器/操作员站/工程师站/被控物理过程/传感器/执行器> | <读/写>       | <提取出的行为内容> | <改变行为客体的输入/改变行为客体的输出/改变行为客体的设定值> | <开关状态/温度设定值/输入电压/其他类似目标参数名称...> | <开/关/调高/调低/其他类似目标参数值...> | <触发该行为的原因> | <周期式/中断式>   | <该行为的结果>   |


### 说明：
1. **行为主体的设备名称**：指执行该行为的设备名称。
2. **行为主体的设备类型**：行为主体的设备类型，如控制器、传感器等。
3. **行为客体的设备名称**：受该行为影响的设备名称。
4. **行为客体的设备类型**：行为客体的设备类型。
5. **行为类型**：如读/写操作。
6. **行为内容**：具体的行为描述。
7. **行为目标类型**：操作目标的类型，例如改变输入、输出或设定值。
8. **行为目标参数名称**：具体的控制参数名称，如“阀门开关状态”、“温度设定值”等。
9. **行为目标参数值**：控制参数的值，如“开”、“关”、“调高”、“调低”等。
10. **行为触发原因**：导致该行为触发的条件或事件。
11. **行为触发类型**：周期式或中断式，描述行为触发的方式。
12. **行为触发结果**：行为执行后的结果描述。

### 输出示例：


| **行为主体的设备名称** | **行为主体的设备类型** | **行为客体的设备名称** | **行为客体的设备类型** | **行为类型** | **行为内容** | **行为目标类型** | **行为目标参数名称** | **行为目标参数值** | **行为触发原因** | **行为触发类型** | **行为触发结果** |
|-----------------------|------------------------|------------------------|------------------------|--------------|--------------|------------------|----------------------|---------------------|-------------------|-------------------|-------------------|
| 控制器                 | 控制器                 | 电动阀门                | 执行器                 | 写           | 发送开启指令   | 改变行为客体的开关状态 | 阀门开关状态           | 开                 | 系统压力超出阈值    | 中断式            | 阀门开启，流量增加，压力降低 |
| 操作员站               | 操作员站               | 电动泵                  | 执行器                 | 写           | 调整功率       | 改变行为客体的功率设定值 | 功率设定值             | 调高               | 流量低于设定值      | 中断式            | 流量恢复，系统稳定  |
| 控制器                 | 控制器                 | 电动泵                  | 执行器                 | 写           | 调整温度       | 改变行为客体的设定值  | 温度设定值             | null            | 温度超出设定范围    | 中断式            | 温度设定调整，恢复正常 |
| 传感器                 | 传感器                 | 控制器                  | 控制器                 | 读           | 读取流量数据   | 改变行为客体的输入    | 流量                   | null            | 流量低于预定值      | 周期式            | 控制器调整输出设定  |
| 工程师站               | 工程师站               | 电动阀门                | 执行器                 | 写           | 修改阀门位置   | 改变行为客体的设定值  | 阀门位置               | 调整               | 未知故障触发      | 中断式            | 阀门位置调整，恢复系统状态 |
| 控制器                 | 控制器                 | 被控物理过程            | 被控物理过程           | 写           | 调节温度       | 改变行为客体的设定值  | 温度设定值             | null            | 温度过高触发       | 中断式            | 温度恢复正常，系统恢复 |



    """
)

# Chinese tokenizer function
def chinese_tokenizer(text):
    return list(jieba.cut(text, cut_all=False))
        
# File upload handler
def save_uploaded_files(uploaded_files):
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

# LLM Configuration Section
st.sidebar.markdown('<div class="section-heading">Language Model Configuration</div>', unsafe_allow_html=True)

llm_col1, llm_col2 = st.sidebar.columns(2)
with llm_col1:
    st.markdown('<div class="parameter-label">Temperature</div>', unsafe_allow_html=True)
    temperature = st.slider("", 0.0, 1.5, 1.0, 0.1, key="temp_slider", label_visibility="collapsed")
    st.markdown(f'<div class="parameter-value">{temperature}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8em;color:#6B7280;text-align:center;">Creativity level</div>', unsafe_allow_html=True)

with llm_col2:
    st.markdown('<div class="parameter-label">Top P</div>', unsafe_allow_html=True)
    top_p = st.slider("", 0.0, 1.0, 0.9, 0.05, key="top_p_slider", label_visibility="collapsed")
    st.markdown(f'<div class="parameter-value">{top_p}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8em;color:#6B7280;text-align:center;">Response variance</div>', unsafe_allow_html=True)

if st.sidebar.button("Update Language Model", use_container_width=True):
    with st.spinner("Updating model..."):
        # Create new LLM instance
        st.session_state.llm_pipeline = LLMPipeline(model_type="deepseek", temperature=temperature, top_p=top_p)
        st.session_state.temperature = temperature
        st.session_state.top_p = top_p
        st.session_state.llm_ready = True
        time.sleep(0.5)  # Provide visual feedback

# KnowledgeBase Configuration Section
st.sidebar.markdown('<div class="section-heading">Knowledge Base Configuration</div>', unsafe_allow_html=True)

# Get all knowledgebase names
knowledgebase_collection = db["knowledgebase"]
collection_names = knowledgebase_collection.find({}, {"collection_name": 1, "_id": 0})
collection_name_list = [doc["collection_name"] for doc in collection_names]

knowledgebase = st.sidebar.selectbox(
    "Select Knowledge Base",
    collection_name_list,
    help="Choose which knowledge base to use for extraction"
)

kb_col1, kb_col2 = st.sidebar.columns(2)
with kb_col1:
    st.markdown('<div class="parameter-label">Semantic Weight</div>', unsafe_allow_html=True)
    semantic_based = st.slider("", 0.0, 1.0, 0.5, 0.05, key="semantic_slider", label_visibility="collapsed")
    st.markdown(f'<div class="parameter-value">{semantic_based}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8em;color:#6B7280;text-align:center;">Meaning-based</div>', unsafe_allow_html=True)

with kb_col2:
    st.markdown('<div class="parameter-label">Keyword Weight</div>', unsafe_allow_html=True)
    keyword_based = 1.0 - semantic_based
    st.slider("", 0.0, 1.0, keyword_based, 0.05, disabled=True, key="keyword_slider", label_visibility="collapsed")
    st.markdown(f'<div class="parameter-value">{keyword_based:.2f}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8em;color:#6B7280;text-align:center;">Term-based</div>', unsafe_allow_html=True)

if st.sidebar.button("Update Knowledge Base", use_container_width=True):
    with st.spinner("Updating knowledge base..."):
        vector_store_manager = VectorStoreManager(
                    vector_store_type="chroma",
                    collection_name=knowledgebase,
                    embedding_model_name="nomic-embed-text",
                    embedding_type="llama"
                )

        retriever = vector_store_manager.as_retriever()

        # Get all data from Chroma and convert to Document format
        chroma_docs = vector_store_manager.vector_store.get()
        bm25_docs = [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(chroma_docs["documents"], chroma_docs["metadatas"])
        ]
        bm25_retriever = BM25Retriever.from_documents(
            bm25_docs,
            tokenizer=chinese_tokenizer
            )
        bm25_retriever.k = 5

        # Initialize ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever],
            weights=[keyword_based, semantic_based]
        )
        st.session_state.retriever = ensemble_retriever
        st.session_state.knowledgebase = knowledgebase
        st.session_state.semantic_based = semantic_based
        st.session_state.keyword_based = keyword_based
        st.session_state.retriever_ready = True
        time.sleep(0.5)  # Provide visual feedback

# Initialize model if not already done
if "llm_pipeline" not in st.session_state:
    llama_pipeline = LLMPipeline(model_type="deepseek")
    st.session_state.llm_pipeline = llama_pipeline
    st.session_state.temperature = temperature
    st.session_state.top_p = top_p
    st.session_state.llm_ready = True

# Initialize retriever if not already done
if "retriever" not in st.session_state:
    vector_store_manager = VectorStoreManager(
                vector_store_type="chroma",
                collection_name="thermal_power",
                embedding_model_name="nomic-embed-text",
                embedding_type="llama"
            )

    retriever = vector_store_manager.as_retriever()

    # Get all data from Chroma and convert to Document format
    chroma_docs = vector_store_manager.vector_store.get()
    bm25_docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(chroma_docs["documents"], chroma_docs["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(
        bm25_docs,
        tokenizer=chinese_tokenizer
        )
    bm25_retriever.k = 5

    # Initialize ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5])
    st.session_state.retriever = ensemble_retriever
    st.session_state.knowledgebase = knowledgebase
    st.session_state.semantic_based = semantic_based
    st.session_state.keyword_based = keyword_based
    st.session_state.retriever_ready = True

# System Status section in sidebar
st.sidebar.markdown('<div class="section-heading">System Status</div>', unsafe_allow_html=True)

status_col1, status_col2 = st.sidebar.columns(2)
with status_col1:
    if st.session_state.llm_ready:
        st.markdown('<div class="status-ready">✓ LLM Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-processing">⚠ LLM Not Ready</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size:0.8em;color:#6B7280;margin-top:5px;">
        Model: DeepSeek<br>
        Temp: {st.session_state.temperature}<br>
        Top-P: {st.session_state.top_p}
    </div>
    """, unsafe_allow_html=True)

with status_col2:
    if st.session_state.retriever_ready:
        st.markdown('<div class="status-ready">✓ KB Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-processing">⚠ KB Not Ready</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size:0.8em;color:#6B7280;margin-top:5px;">
        KB: {st.session_state.knowledgebase}<br>
        Semantic: {st.session_state.semantic_based}<br>
        Keyword: {st.session_state.keyword_based:.2f}
    </div>
    """, unsafe_allow_html=True)

# Retrieve function
def retrieve(query, retriever):
    result = retriever.invoke(query)
    return {"context": result, "question": query}

# Main content area
st.markdown('<div class="stcard">', unsafe_allow_html=True)
st.markdown('<div class="section-heading">Upload Operational Specifications</div>', unsafe_allow_html=True)

# Enhanced file upload section
# st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
uploaded_files = st.file_uploader("", accept_multiple_files=True, label_visibility="collapsed")

if not uploaded_files:
    # Show placeholder content when no files uploaded
    st.markdown("""
        <div style="text-align:center">
            <svg xmlns="http://www.w3.org/2000/svg" width="50" height="50" fill="#3B82F6" viewBox="0 0 16 16">
                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                <path d="M8 4a.5.5 0 0 1 .5.5v3h3a.5.5 0 0 1 0 1h-3v3a.5.5 0 0 1-1 0v-3h-3a.5.5 0 0 1 0-1h3v-3A.5.5 0 0 1 8 4z"/>
            </svg>
            <p style="margin-top:10px;font-weight:500;color:#334155;">Drop PDF files here or click to browse</p>
            <p style="color:#6B7280;font-size:0.9em;">Supported format: PDF</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Display file information
    st.markdown('<div class="file-list">', unsafe_allow_html=True)
    for file in uploaded_files:
        # Calculate file size
        file_size = round(file.size / 1024, 1)
        file_ext = os.path.splitext(file.name)[1].lower()
        
        st.markdown(f"""
            <div class="file-item">
                <div class="file-icon">📄</div>
                <div class="file-name">{file.name}</div>
                <div class="file-size">{file_size} KB</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process files when uploaded
    if 'data_value' not in st.session_state or not st.session_state.data_value:
        with st.spinner("Processing uploaded files..."):
            session_folder, data_value = save_uploaded_files(uploaded_files)
            st.session_state["session_folder"] = session_folder
            st.session_state["data_value"] = data_value

            # File processing
            result = []
            for file in st.session_state.data_value:
                reader = PdfReader(file)
                parts = []

                # Remove header and footer
                def visitor_body(text, cm, tm, fontDict, fontSize):
                    y = tm[5]
                    if y > 70 and y < 770:
                        parts.append(text)

                for i in range(0, len(reader.pages), 3):
                    if i < len(reader.pages) - 4:
                        reader.pages[i].extract_text(visitor_text=visitor_body)
                        reader.pages[i+1].extract_text(visitor_text=visitor_body)
                        reader.pages[i+2].extract_text(visitor_text=visitor_body)
                        reader.pages[i+3].extract_text(visitor_text=visitor_body)
                        result.append("".join(parts))
                        parts.clear()
                    else:
                        reader.pages[len(reader.pages)-3].extract_text(visitor_text=visitor_body)
                        reader.pages[len(reader.pages)-2].extract_text(visitor_text=visitor_body)
                        reader.pages[len(reader.pages)-1].extract_text(visitor_text=visitor_body)
                        result.append("".join(parts))
                        parts.clear()
                        break
            
            st.session_state["raw_text"] = result
            st.session_state["length"] = len(result)
            
            st.markdown(f"""
            <div class="success-message">
                <strong>Files processed successfully!</strong> The document has been split into {len(result)} segments for analysis.
            </div>
            """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state variables
if 'results' not in st.session_state:
    st.session_state.results = []
if "current_page" not in st.session_state:
    st.session_state.current_page = 0
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

# Extraction Button
if 'data_value' in st.session_state and st.session_state.data_value:
    st.markdown("""
    <button class="primary-button" id="extract-button">
        <div style="display:flex;align-items:center;justify-content:center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" style="margin-right:8px;" viewBox="0 0 16 16">
                <path d="M5.52.359A.5.5 0 0 1 6 0h4a.5.5 0 0 1 .474.658L8.694 6H12.5a.5.5 0 0 1 .395.807l-7 9a.5.5 0 0 1-.873-.454L6.823 9.5H3.5a.5.5 0 0 1-.48-.641l2.5-8.5z"/>
            </svg>
            Start Extraction Process
        </div>
    </button>
    <script>
        document.getElementById("extract-button").addEventListener("click", function() {
            window.parent.postMessage({type: "streamlit:buttonClicked", data: "extract-button-id"}, "*");
        });
    </script>
    """, unsafe_allow_html=True)
    
    if st.button("Start Extraction Process", key="extract-button-id", use_container_width=True):
        st.session_state.clicked = True

# Extraction Process
if st.session_state.clicked and 'data_value' in st.session_state and st.session_state.data_value:
    # Start time
    st.session_state.start_time = datetime.now()
    # Result filename
    unique_filename = f"result_{uuid.uuid4().hex}.md"
    st.session_state.result_file = f"result/{unique_filename}"
    
    # Create directory if it doesn't exist
    os.makedirs("result", exist_ok=True)
    
    # Create progress display
    progress_container = st.empty()
    with progress_container.container():
        st.markdown('<div class="section-heading">Extraction Progress</div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        progress_text = st.empty()
        progress_text.markdown("Preparing extraction process...")
    
    # Total steps calculation
    progress_total = st.session_state.length * 3
    progress_current = 0
    
    # Results display
    results_container = st.container()
    with results_container:
        st.markdown('<div class="section-heading">Extraction Results</div>', unsafe_allow_html=True)
        
        # Create tabs for different result types
        device_tab, topology_tab, logic_tab = st.tabs(["Device Information", "Business Topology", "Control Logic"])
        
        with device_tab:
            device_placeholder = st.empty()
        
        with topology_tab:
            topology_placeholder = st.empty()
        
        with logic_tab:
            rule_placeholder = st.empty()
    
    # Process each segment
    for i in range(st.session_state.length):
        # Update progress
        progress_text.markdown(f"Processing segment {i+1} of {st.session_state.length}...")
        
        # Step 1: Extract device information
        progress_text.markdown(f"Extracting device information from segment {i+1}...")
        retrieved_data = retrieve(st.session_state.raw_text[i], st.session_state.retriever)
        formatted_prompt = chat_prompt_1.format(
            context=retrieved_data["context"],
            question=retrieved_data["question"]
        )
        response = st.session_state.llm_pipeline.run(formatted_prompt)
        temp_result_1 = response['response']['result'].content
        
        # Update device tab
        device_content = device_placeholder.markdown(f"""
        <div class="result-panel">
            <div class="panel-title">Device Information (Segment {i+1})</div>
            <div class="panel-content">{temp_result_1}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Update progress
        progress_current += 1
        progress_bar.progress(progress_current/progress_total)
        progress_text.markdown(f"Progress: {progress_current/progress_total*100:.1f}% - Extracting business topology...")
        
        # Step 2: Extract business topology
        formatted_prompt = chat_prompt_2.format(
            text=st.session_state.raw_text[i],
            device=temp_result_1
        )
        response = st.session_state.llm_pipeline.run(formatted_prompt)
        temp_result_2 = response['response']['result'].content
        
        # Update topology tab
        topology_content = topology_placeholder.markdown(f"""
        <div class="result-panel">
            <div class="panel-title">Business Topology (Segment {i+1})</div>
            <div class="panel-content">{temp_result_2}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Update progress
        progress_current += 1
        progress_bar.progress(progress_current/progress_total)
        progress_text.markdown(f"Progress: {progress_current/progress_total*100:.1f}% - Extracting control logic...")
        
        # Step 3: Extract control logic
        formatted_prompt = chat_prompt_3.format(
            text=st.session_state.raw_text[i],
            action=temp_result_2
        )
        response = st.session_state.llm_pipeline.run(formatted_prompt)
        final_result = response['response']['result'].content
        
        # Update logic tab
        rule_content = rule_placeholder.markdown(f"""
        <div class="result-panel">
            <div class="panel-title">Control Logic (Segment {i+1})</div>
            <div class="panel-content">{final_result}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Write to file
        with open(f"result/{unique_filename}", "a", encoding="utf-8") as file:
            file.write(f"# Segment {i+1} Control Logic\n\n")
            file.write(str(final_result) + "\n\n---\n\n")
        
        # Update progress
        progress_current += 1
        progress_bar.progress(progress_current/progress_total)
        progress_text.markdown(f"Progress: {progress_current/progress_total*100:.1f}%")
    
    # End time
    st.session_state.end_time = datetime.now()
    
    # Insert data to MongoDB
    data = {
        "start_time": st.session_state.start_time,
        "end_time": st.session_state.end_time,
        "source_specification": st.session_state.data_value,
        "pieces": st.session_state.length,
        "LLM_parameters": {
            "name": "gpt-4o-mini",
            "temperature": st.session_state.temperature,
            "top_p": st.session_state.top_p
        },
        "Knowledgebase_parameters": {
            "knowledgebase_name": st.session_state.knowledgebase,
            "semantic_based": st.session_state.semantic_based,
            "keyword_based": st.session_state.keyword_based,
            "embedding_model_name": "nomic-embed-text"
        },
        "result_file": st.session_state.result_file
    }
    
    chats_collection.insert_one(data)
    
    # Update progress indicator
    progress_bar.progress(1.0)
    progress_text.markdown("Extraction completed successfully!")
    
    # Display completion message
    st.markdown(f"""
    <div class="success-message">
        <strong>Extraction completed!</strong><br>
        Started: {st.session_state.start_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
        Completed: {st.session_state.end_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
        Processing time: {(st.session_state.end_time - st.session_state.start_time).total_seconds():.2f} seconds<br>
        Result saved to: {st.session_state.result_file}
    </div>
    """, unsafe_allow_html=True)
    
    # Add download button
    with open(st.session_state.result_file, "r", encoding="utf-8") as file:
        file_content = file.read()
        st.download_button(
            label="Download Results",
            data=file_content,
            file_name=os.path.basename(st.session_state.result_file),
            mime="text/markdown",
            key="download_button"
        )
    
    # Reset states for next extraction
    st.session_state.clicked = False
    st.session_state["session_folder"] = ''
    st.session_state["data_value"] = ''
    st.session_state["raw_text"] = ''
    st.session_state["length"] = ''

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; color: #6B7280; font-size: 0.8em; border-top: 1px solid #e5e7eb;">
    Interaction Logic Extraction System © 2025 | Version 1.0
</div>
""", unsafe_allow_html=True)














# import streamlit as st
# from pymongo import MongoClient
# from src.run_rag_pipeline import RunChatbot
# import os
# import json
# import uuid
# from dotenv import load_dotenv
# import json
# import uuid
# import logging
# import time 
# from io import StringIO
# import shutil  # 导入用于文件夹清理的模块
# import requests
# from streamlit_lottie import st_lottie
# from src.llm.llm_pipeline import LLMPipeline
# from PyPDF2 import PdfReader
# from src.llm.llm_pipeline import LLMPipeline
# from src.rag.vector_store import VectorStoreManager
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain.docstore.document import Document
# from langchain.prompts import PromptTemplate
# import jieba
# import math
# from datetime import datetime



# # 直接写死 MongoDB URI
# mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# # 连接到 MongoDB
# mongo_client = MongoClient(mongo_url)

# # 访问数据库和集合
# db = mongo_client["rag"]  # 使用数据库 rag
# chats_collection = db["history"]  # 使用集合 history

# # Streamlit 应用标题
# st.title("Interaction Logic Extraction")


# # CSS 样式
# def apply_styles():
#     st.markdown("""
#     <style>
#         /* General Body Styling */
#         body {
#             background-color: #f2f2f2; /* Light grey background */
#             font-family: 'Arial', sans-serif;
#             color: black; /* Standard black text for contrast */
#         }

#         /* Sidebar Customization */
#         section[data-testid="stSidebar"] {
#             background-color: #e6e6e6; /* Light grey sidebar */
#             color: black; /* Black text for readability */
#         }
#         section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
#             color: #333; /* Dark grey for headings in the sidebar */
#         }

#         /* Input and Button Styles */
#         .stTextInput, .stTextArea, .stSelectbox, .stRadio {
#             background-color: #ffffff !important; /* White background for inputs */
#             border: 1px solid #ccc !important; /* Subtle border */
#             border-radius: 5px !important;
#             padding: 10px !important;
#             color: black; /* Black text */
#         }

#         .stButton button {
#             background-color: #4caf50 !important; /* Subtle green for buttons */
#             color: white !important;
#             border: none;
#             border-radius: 5px;
#             padding: 10px;
#             font-size: 14px;
#             cursor: pointer;
#             transition: all 0.3s ease-in-out;
#         }

#         .stButton button:hover {
#             background-color: #45a049 !important; /* Slightly darker green on hover */
#         }

#         /* Header Styling */
#         h1, h2, h3, h4 {
#             color: #333; /* Dark grey for headers */
#             font-weight: 600;
#         }

#         /* Chat Bubble Styles */
#         .chat-container {
#             display: flex;
#             flex-direction: column;
#             gap: 10px;
#             padding: 10px;
#         }

#         .chat-bubble {
#             max-width: 80%;
#             padding: 12px 18px;
#             border-radius: 10px;
#             margin-bottom: 10px;
#         }

#         .user-message {
#             background-color: #d9fdd3; /* Light green for user messages */
#             color: black;
#             align-self: flex-end;
#         }

#         .assistant-message {
#             background-color: #e6e6e6; /* Light grey for assistant messages */
#             color: black;
#             align-self: flex-start;
#         }

#         /* Footer Styling */
#         footer {
#             text-align: center;
#             font-size: 12px;
#             color: #666;
#             margin-top: 50px;
#         }
#     </style>
#     """, unsafe_allow_html=True)

# # 加载 Lottie 动画
# def load_lottie_url(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# # 应用样式
# apply_styles()

# # 添加 Lottie 动画
# def add_header_animation():
#     lottie_url = "https://assets9.lottiefiles.com/packages/lf20_ksu5dpjr.json"  # Clean chatbot animation
#     animation_data = load_lottie_url(lottie_url)
#     if animation_data:
#         st_lottie(animation_data, height=200, key="header_animation")

# add_header_animation()


# # 带交互选项的侧边栏
# st.sidebar.markdown("""
# <div style="
#     background-color: #585858; 
#     padding: 5px; 
#     border-radius: 5px; 
#     text-align: center; 
#     margin-bottom: 5px;
#     border: 1px solid #dcdcdc;">
#     <h2 style="color: white; font-family: 'Arial', sans-serif; margin: 0;">
#         <strong>Customization Options</strong> 
#     </h2>
# </div>
# """, unsafe_allow_html=True)


    
# # 日志记录器设置
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)





# chat_prompt_1 = PromptTemplate(
#     input_variables = ["context","question"],
#     template = """
            
#             这是**工控系统文本**，**所有分析都基于此文本**：{question}
#             这是检索到的相关信息，**如果存在工控系统文本中不存在的内容请忽略**：{context}

#             ## 角色简介:
#             - 角色: 工控系统单点设备与属性提取大师
#             - 语言: 中文
#             - 描述:专注于工控系统中单点设备的识别与属性提取，精准分析各设备的名称、定位与类型，确保完整理解系统中的设备结构，为下一步的业务拓扑的提取做准备。

#             ## 核心目标:
#             - 第一步，明确工控系统文本中单点设备识别的目标和范围。你需要根据目标工控系统的特点，分析系统中涉及控制、监测和执行的单点设备，如控制器、操作员站、工程师站、被控物理过程、传感器和执行器等，为后续属性提取奠定基础。
#             - 第二步，分析各设备的设备名称，此名称是唯一的。
#             - 第三步，分析设备在该段文本中的设备定位，定位是不唯一的，可能是行为主体也可能是行为客体。
#             - 第四步，分析设备的设备类型，可能是控制器、操作员站、工程师站、被控物理过程、传感器或执行器，设备类型是唯一的。
#             - 第五步，结合前四步的分析，系统化提取设备的核心属性，并以清晰的结构进行整理，确保数据的可用性与可操作性。

#             ## 约束条件:
#             - 推理过程必须逻辑严密。每一步分析应具备层层递进的逻辑关系，确保设备属性提取的准确性与完整性。
#             - 设备识别需精确。避免误分类或遗漏，确保所有核心设备都被识别并提取。
#             - 过程模型分析需深入，明确设备的设备名称、设备定位及设备类型。
#             - 避免遗漏细节，确保数据的完整性和可操作性。

#             ## 重要补充：
#             - **如果检索到的相关信息（{context}）与任务无关或无助于提取设备属性，可以忽略该信息，仅基于工控系统文本（{question}）进行分析**。
#             - 如果某些设备属性无法从工控系统文本中提取出来，则在输出格式中填充“不存在”。不要编造信息。
#             - 如果工控系统文本中不存在任何可提取的单点设备，则输出“【不存在单点设备】”。
#             - 如果工控系统文本中存在可提取的单点设备，则必须完整提取每一个设备，并按照规定的格式输出，不得遗漏。

#             ## 专项技能:
#             - 精准识别单点设备: 能够从复杂的工控系统文本中精准识别控制器、操作员站、工程师站、被控物理过程、传感器和执行器等设备，并准确归类。
#             - 系统化逻辑提取能力: 能够从多个角度分析设备的属性，并以清晰的结构整理输出，确保数据的高效利用。
#             - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别设备及其属性，确保最终提取的逻辑严谨且无遗漏。

#             ## 处理流程:
#             - 作为工控系统单点设备与属性提取大师，你将**以提供的工控系统文本为主体**，并结合检索到的相关信息作为辅助，运用你的“专项技能”能力分析设备名称、设备定位和设备类型。
#             - 第一步，你需要一步步思考并推理，分析提供的工控系统文本和检索到的相关信息，分析检索到的相关信息中是否存在不属于提供的工控系统文本中的内容，如果存在，忽略检索到的相关信息中的内容。
#                 例如：火焰切割机在检索到的相关信息中出现过，但提供的工控系统文本中没有提及过火焰收割机，则忽略火焰收割机的相关内容
#             - 第二步，你需要一步步思考并推理，从**工控系统文本中**识别出所有的单点设备。
#             - 第三步，你需要一步步思考并推理，详细分析各个单点设备的设备名称，此名称是唯一的，确保设备输入输出的完整描述。
#             - 第四步，你需要一步步思考并推理，详细分析各个单点设备的设备类型，此类型是唯一的，可能是控制器、操作员站、工程师站、被控物理过程、传感器或执行器。
#             - 第五步，你需要一步步思考并推理，结合以上四步你的推理过程，综合整理设备属性信息，确保每一个提取出的设备都能在提供的工控系统文本中找到，同时系统化提取并确保数据完整，并且将提取出的单点设备按输出格式表述出来。

#             ## 输出格式:
#             【推理过程】<该设备的提取过程：200字>
#             【设备名称】<提取出的设备名称>
#             【设备定位】<行为主体/行为客体/行为主体和行为客体>
#             【设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>


#     """
# )


# chat_prompt_2 = PromptTemplate(
#     input_variables = ["device","text"],
#     template = """

#         这是**工控系统文本**，**所有分析都基于此文本**：{text}
#         这是已经提取出的单点设备：{device}

#         ## 角色简介:
#         - 角色: 工控系统业务拓扑提取大师
#         - 语言: 中文
#         - 描述: 专注于工控系统中的业务拓扑结构提取，精准识别系统中行为主体和行为客体之间发生的行为内容，并细致分析行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值，确保全面理解系统的控制与反馈机制，为后续行为上下文的提取提供关键数据支持。
       
#         ## 核心目标:
#         - 第一步，明确业务拓扑分析的目标和范围。你需要从工控系统文本中识别各类控制行为，包括控制指令、数据读取/写入、反馈机制等，为后续分析奠定基础，同时结合已经提取出的单点设备，明确每个控制行为的行为主体和行为客体。
#         - 第二步，深入分析各类行为的行为类型，行为类型为读或者写，行为类型是唯一的。
#         - 第三步，分析行为的行为目标类型，行为目标类型为改变行为客体的输入、改变行为客体的输出或改变行为客体的设定值，行为目标类型是唯一的。
#         - 第四步，分析行为的行为目标参数名称，行为目标参数名称可以是开关状态、温度设定值、输入电压或者其他类似目标参数名称，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
#         - 第五步，分析行为的行为目标参数值，行为目标参数值可以是开、关、调高、调低或者其他类似目标参数值，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
#         - 第六步，结合前五步的分析，系统化提取业务拓扑数据，并整理输出，以支持系统优化与控制策略制定。

#         ## 约束条件:
#         - 推理过程必须逻辑严密。每一步分析应具备层层递进的逻辑关系，确保业务拓扑提取的准确性与完整性。
#         - 业务拓扑识别需精确。避免误分类或遗漏，确保所有业务拓扑都被识别并提取。
#         - 推理分析需深入，明确行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值。
#         - 行为目标参数名称不可以缺少，如果提取不出来，可以将其值设置为null。
#         - 行为目标参数值不可以缺少，如果提取不出来，可以将其值设置为null。
#         - 避免遗漏细节，确保数据的完整性和可操作性。

#         ## 重要补充：
#         - 如果某些业务拓扑的相关属性数据无法从工控系统文本中提取出来，则在输出格式中填充“不存在”。不要编造信息。
#         - 如果工控系统文本中不存在任何可提取的业务拓扑结构，则输出“【不存在业务拓扑】”。
#         - 如果工控系统文本中存在可提取的业务拓扑结构，则必须完整提取每一个业务拓扑结构，并按照规定的格式输出，不得遗漏。

#         ## 专项技能:
#         - 精准识别业务行为: 能够从复杂的工控系统文本中精准提取业务拓扑，确保分析全面。
#         - 系统化拓扑提取能力: 能够从多个角度分析行为的相互关系，并以清晰的结构整理输出，确保数据的高效利用。
#         - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别业务行为及其属性，确保最终提取的逻辑严谨且无遗漏。

#         ## 处理流程:
#         - 作为工控系统业务拓扑提取大师，你将从提供的工控系统文本中，结合已经提取出的单点设备，运用你的“专项技能”能力精准识别系统中行为主体和行为客体之间发生的行为内容，并细致分析行为内容的行为类型、行为内容、行为目标类型、行为目标参数名称和行为目标参数值
#         - 第一步，你需要一步步思考并推理，从文本中识别业务行为，并明确它们的作用，同时结合已经提取出的单点设备，明确每个业务行为的行为主体和行为客体。
#         - 第二步，你需要一步步思考并推理，各类行为的行为类型，行为类型为读或者写，行为类型是唯一的，确保行为的精准描述。
#         - 第三步，你需要一步步思考并推理，分析行为的行为目标类型，行为目标类型为改变行为客体的输入、改变行为客体的输出或改变行为客体的设定值，行为目标类型是唯一的。
#         - 第四步，你需要一步步思考并推理，分析行为的行为目标参数名称，行为目标参数名称可以是开关状态、温度设定值、输入电压或这其他类似目标参数名称，具体要看文本中的行为是什么样的如果提取不出来，则设置为null。
#         - 第五步，你需要一步步思考并推理，分析行为的行为目标参数值，行为目标参数值可以是开、关、调高、调低或者其他类似目标参数值，具体要看文本中的行为是什么样的，如果提取不出来，则设置为null。
#         - 第六步，你需要一步步思考并推理，结合前五步的分析，系统化提取业务拓扑数据，并整理输出，以支持系统优化与控制策略制定，并且要把提取出的业务拓扑按照输出格式进行输出。

    
#         ## 输出格式:
    
#         【行为主体的设备名称】<设备名称>
#         【行为主体的设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>

#         【行为客体的设备名称】<设备名称>
#         【行为客体的设备类型】<控制器/操作员站/工程师站/被控物理过程/传感器/执行器>

#         【行为类型】<读/写>
#         【行为内容】<提取出的行为内容>
#         【行为目标类型】<改变行为客体的输入/改变行为客体的输出/改变行为客体的设定值>
#         【行为目标参数名称】<开关状态/温度设定值/输入电压/其他类似目标参数名称.../null>
#         【行为目标参数值】<开/关/调高/调低/其他类似目标参数值.../null>


#     """
# )

# chat_prompt_3 = PromptTemplate(
#     input_variables = ["text","action"],
#     template = """

#     这是**工控系统文本**，**所有分析都基于此文本**：{text}
#     这是已经提取出的业务拓扑：{action}

#     ## 角色简介:
#     - 角色: 工控系统业务逻辑提取大师
#     - 语言: 中文
#     - 描述: 专注于工控系统中的业务（行为）逻辑提取，精准分析行为的触发原因、触发类型及触发结果，确保完整理解系统的控制流程和反馈机制，为优化系统控制策略提供关键数据支持。

#     ## 核心目标:
#     - 第一步，明确业务逻辑分析的目标和范围。你需要结合工控系统文本和已经提取出的业务拓扑，识别每个业务拓扑中业务行为的触发原因、触发类型及触发结果。 
#     - 第二步，识别行为的触发原因，是什么导致了该行为的发生。
#     - 第二步，识别行为的触发类型，明确系统中的业务逻辑是周期式（定期执行的控制行为）还是中断式（由事件或特定条件触发的行为），触发类型唯一。
#     - 第三步，识别行为的触发结果，该行为导致了什么结果。
#     - 第五步，结合前三步的分析，系统化提取业务逻辑，并整理输出，以支持系统优化与控制策略制定。

#     ## 约束条件:
#     - 推理过程必须逻辑严密。确保对每个行为的原因和结果分析层层递进，保证业务逻辑的完整性和准确性。
#     - 触发类型识别需精准。确保正确区分周期式和中断式触发方式，避免误分类。
#     - 确保业务逻辑的可操作性。所有提取的信息需具有实际指导意义，确保可用于优化控制策略。
#     - 如果按照输出格式表述的时候，有表格对应参数不存在或者提取不来，用null进行占位，不要空着



#     ## 专项技能:
#     - 精准识别行为的触发类型: 能够从复杂的工控系统文本中精准提取周期式触发和中断式触发的业务逻辑，确保逻辑分析全面准确。
#     - 系统化逻辑提取能力: 能够从多个角度分析行为的上下文，并以清晰的结构整理输出，确保数据的高效利用。
#     - 细致的推理与逻辑梳理: 通过分步推理，层层递进地识别业务逻辑及其触发机制，确保最终提取的逻辑严谨且无遗漏。

#     ## 处理流程:
#     - 作为工控系统业务逻辑提取大师，你将从提供的工控系统文本中，结合已经提取出的业务拓扑，运用你的“专项技能”能力分析行为上下文的触发原因、触发类型及触发结果。
#     - 第一步，你需要一步步思考并推理，从文本中识别行为的触发原因，是什么导致了该行为的发生。
#     - 第二步，你需要一步步思考并推理，详细识别行为的触发类型，明确系统中的业务逻辑是周期式（定期执行的控制行为）还是中断式（由事件或特定条件触发的行为），触发类型是唯一的。
#     - 第三步，你需要一步步思考并推理，识别行为的触发结果，该行为导致了什么结果。
#     - 第四步，你需要一步步思考并推理，结合以上三步你的推理过程，综合整理业务逻辑数据，系统化提取并确保数据完整，以便后续优化系统控制逻辑，并且将提取出的业务逻辑按输出格式表述出来。
#     请注意：**如果按照输出格式表述的时候，有表格对应参数不存在或者提取不来，用null进行占位，不要空着**

    

#     ## 输出格式:
# | **行为主体的设备名称** | **行为主体的设备类型** | **行为客体的设备名称** | **行为客体的设备类型** | **行为类型** | **行为内容** | **行为目标类型** | **行为目标参数名称** | **行为目标参数值** | **行为触发原因** | **行为触发类型** | **行为触发结果** |
# |-----------------------|------------------------|------------------------|------------------------|--------------|--------------|------------------|----------------------|---------------------|-------------------|-------------------|-------------------|
# | <设备名称>            | <控制器/操作员站/工程师站/被控物理过程/传感器/执行器> | <设备名称>            | <控制器/操作员站/工程师站/被控物理过程/传感器/执行器> | <读/写>       | <提取出的行为内容> | <改变行为客体的输入/改变行为客体的输出/改变行为客体的设定值> | <开关状态/温度设定值/输入电压/其他类似目标参数名称...> | <开/关/调高/调低/其他类似目标参数值...> | <触发该行为的原因> | <周期式/中断式>   | <该行为的结果>   |


# ### 说明：
# 1. **行为主体的设备名称**：指执行该行为的设备名称。
# 2. **行为主体的设备类型**：行为主体的设备类型，如控制器、传感器等。
# 3. **行为客体的设备名称**：受该行为影响的设备名称。
# 4. **行为客体的设备类型**：行为客体的设备类型。
# 5. **行为类型**：如读/写操作。
# 6. **行为内容**：具体的行为描述。
# 7. **行为目标类型**：操作目标的类型，例如改变输入、输出或设定值。
# 8. **行为目标参数名称**：具体的控制参数名称，如“阀门开关状态”、“温度设定值”等。
# 9. **行为目标参数值**：控制参数的值，如“开”、“关”、“调高”、“调低”等。
# 10. **行为触发原因**：导致该行为触发的条件或事件。
# 11. **行为触发类型**：周期式或中断式，描述行为触发的方式。
# 12. **行为触发结果**：行为执行后的结果描述。

# ### 输出示例：


# | **行为主体的设备名称** | **行为主体的设备类型** | **行为客体的设备名称** | **行为客体的设备类型** | **行为类型** | **行为内容** | **行为目标类型** | **行为目标参数名称** | **行为目标参数值** | **行为触发原因** | **行为触发类型** | **行为触发结果** |
# |-----------------------|------------------------|------------------------|------------------------|--------------|--------------|------------------|----------------------|---------------------|-------------------|-------------------|-------------------|
# | 控制器                 | 控制器                 | 电动阀门                | 执行器                 | 写           | 发送开启指令   | 改变行为客体的开关状态 | 阀门开关状态           | 开                 | 系统压力超出阈值    | 中断式            | 阀门开启，流量增加，压力降低 |
# | 操作员站               | 操作员站               | 电动泵                  | 执行器                 | 写           | 调整功率       | 改变行为客体的功率设定值 | 功率设定值             | 调高               | 流量低于设定值      | 中断式            | 流量恢复，系统稳定  |
# | 控制器                 | 控制器                 | 电动泵                  | 执行器                 | 写           | 调整温度       | 改变行为客体的设定值  | 温度设定值             | null            | 温度超出设定范围    | 中断式            | 温度设定调整，恢复正常 |
# | 传感器                 | 传感器                 | 控制器                  | 控制器                 | 读           | 读取流量数据   | 改变行为客体的输入    | 流量                   | null            | 流量低于预定值      | 周期式            | 控制器调整输出设定  |
# | 工程师站               | 工程师站               | 电动阀门                | 执行器                 | 写           | 修改阀门位置   | 改变行为客体的设定值  | 阀门位置               | 调整               | 未知故障触发      | 中断式            | 阀门位置调整，恢复系统状态 |
# | 控制器                 | 控制器                 | 被控物理过程            | 被控物理过程           | 写           | 调节温度       | 改变行为客体的设定值  | 温度设定值             | null            | 温度过高触发       | 中断式            | 温度恢复正常，系统恢复 |



#     """
# )

# # 使用jieba进行分词
# def chinese_tokenizer(text):
#     return list(jieba.cut(text, cut_all=False))
        
# def save_uploaded_files(uploaded_files):
#     """Save uploaded files and return their paths."""
#     if not uploaded_files:
#         return []
#     session_folder = f"uploads/{uuid.uuid4()}"
#     os.makedirs(session_folder, exist_ok=True)
#     file_paths = []
#     for uploaded_file in uploaded_files:
#         file_path = os.path.join(session_folder, uploaded_file.name)
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         file_paths.append(file_path)
#     return session_folder, file_paths



# # 侧边栏：大模型参数调节
# st.sidebar.header("Large Language Model Parameters")

# temperature = st.sidebar.slider("Choose temperature", 0.0, 1.5, 1.0)

# top_p = st.sidebar.slider("Choose top_p", 0.0, 1.0, 0.9)
     
# if st.sidebar.button("Update LLM"):
#     # 创建新的 LLM 实例
#     st.session_state.llm_pipeline = LLMPipeline(model_type="deepseek", temperature=temperature, top_p=top_p)
#     st.session_state.temperature = temperature
#     st.session_state.top_p = top_p
#     st.session_state.llm_ready = True




# # 获得所有知识库名称
# knowledgebase_collection = db["knowledgebase"]  # 使用集合 knowledgebase
# # 获取所有 collection_name 字段
# collection_names = knowledgebase_collection.find({}, {"collection_name": 1, "_id": 0})
# # 提取 collection_name 字段
# collection_name_list = [doc["collection_name"] for doc in collection_names]
# # 侧边栏：向量知识库参数调节
# st.sidebar.header("Vector KnowledgeBase Parameters")

# knowledgebase = st.sidebar.selectbox(
#     "Select KnowledgeBase",
#     collection_name_list
# )

# semantic_based = st.sidebar.slider("Choose The proportion of semantic-based retrieval", 0.0, 1.0, 0.5)
# keyword_based = st.sidebar.slider("The proportion of keyword-based retrieval", 0.0, 1.0, 1.0 - semantic_based, disabled = True)

# if st.sidebar.button("Update KnowledgeBase"):
#     vector_store_manager = VectorStoreManager(
#                 vector_store_type="chroma",
#                 collection_name= knowledgebase,
#                 embedding_model_name = "nomic-embed-text",
#                 #embedding_model_name="llama3",
#                 embedding_type="llama"
#             )

#     retriever = vector_store_manager.as_retriever()


#     # 从Chroma 取出所有数据，并转化为Document 格式
#     chroma_docs = vector_store_manager.vector_store.get()
#     bm25_docs = [
#         Document(page_content=text, metadata = metadata)
#         for text, metadata in zip(chroma_docs["documents"],chroma_docs["metadatas"])
#     ]
#     bm25_retriever = BM25Retriever.from_documents(
#         bm25_docs,
#         tokenizer = chinese_tokenizer # 显式指定中文分词器
#         )
#     bm25_retriever.k = 5

#     # 初始化集成检索器
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],weights=[keyword_based, semantic_based])
#     st.session_state.retriever = ensemble_retriever
#     st.session_state.knowledgebase = knowledgebase
#     st.session_state.semantic_based = semantic_based
#     st.session_state.keyword_based = keyword_based
#     st.session_state.retriever_ready = True
   




# # 模型准备
# if "llm_pipeline" not in st.session_state:
#     llama_pipeline = LLMPipeline(model_type="deepseek" )
#     st.session_state.llm_pipeline = llama_pipeline
#     st.session_state.temperature = temperature
#     st.session_state.top_p = top_p
#     st.session_state.llm_ready = True



# if st.session_state.llm_ready:
#     st.success(f"LLM has been ready with temperature: {st.session_state.temperature} and top_p: {st.session_state.top_p}.")



# # 检索器设置



# if "retriever" not in st.session_state:
#     vector_store_manager = VectorStoreManager(
#                 vector_store_type="chroma",
#                 collection_name="thermal_power",
#                 embedding_model_name = "nomic-embed-text",
#                 #embedding_model_name="llama3",
#                 embedding_type="llama"
#             )

#     retriever = vector_store_manager.as_retriever()


#     # 从Chroma 取出所有数据，并转化为Document 格式
#     chroma_docs = vector_store_manager.vector_store.get()
#     bm25_docs = [
#         Document(page_content=text, metadata = metadata)
#         for text, metadata in zip(chroma_docs["documents"],chroma_docs["metadatas"])
#     ]
#     bm25_retriever = BM25Retriever.from_documents(
#         bm25_docs,
#         tokenizer = chinese_tokenizer # 显式指定中文分词器
#         )
#     bm25_retriever.k = 5

#     # 初始化集成检索器
#     ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],weights=[0.5,0.5])
#     st.session_state.retriever = ensemble_retriever
#     st.session_state.knowledgebase = knowledgebase
#     st.session_state.semantic_based = semantic_based
#     st.session_state.keyword_based = keyword_based
#     st.session_state.retriever_ready = True

# if st.session_state.retriever_ready:
#     st.success(f"Retriever has been ready with KnowledgeBase: {st.session_state.knowledgebase}, semantic-based: {st.session_state.semantic_based} and keyword-based: {st.session_state.keyword_based}.")


# # 检索函数
# def retrieve(query, retriever):
#     result = retriever.invoke(query)
#     return {"context":result,"question":query}

# # 上传文件(当前只对单个pdf格式文件进行了适配)
# uploaded_files = st.file_uploader("Upload Operational Specifications", accept_multiple_files=True)

# if uploaded_files:
#     session_folder, data_value = save_uploaded_files(uploaded_files)
#     st.session_state["session_folder"] = session_folder
#     st.session_state["data_value"] = data_value

#     # 文件分割
#     for file in st.session_state.data_value:
#         reader = PdfReader(file)
#         parts = []
#         result= []

#         # 去除页头和页尾
#         def visitor_body(text, cm, tm, fontDict, fontSize):
#             y = tm[5]
#             if y>70 and y<770:
#                 parts.append(text)

#         for i in range(0,len(reader.pages),3):
#             if(i<len(reader.pages)-4):
#                 reader.pages[i].extract_text(visitor_text = visitor_body)
#                 reader.pages[i+1].extract_text(visitor_text = visitor_body)
#                 reader.pages[i+2].extract_text(visitor_text = visitor_body)
#                 reader.pages[i+3].extract_text(visitor_text = visitor_body)        
#                 result.append("".join(parts))
#                 parts.clear()
#             else:
#                 reader.pages[len(reader.pages)-3].extract_text(visitor_text = visitor_body)
#                 reader.pages[len(reader.pages)-2].extract_text(visitor_text = visitor_body)
#                 reader.pages[len(reader.pages)-1].extract_text(visitor_text = visitor_body)
#                 result.append("".join(parts))
#                 parts.clear()
#                 break
        
#         st.session_state["raw_text"] = result
#         st.session_state["length"] = len(result)
#         st.write(f"The Operational Specification has been split into {len(result)} pieces.")




# # 初始化存储结果
# if 'results' not in st.session_state:
#     st.session_state.results = []


# # 初始化页码
# if "current_page" not in st.session_state:
#     st.session_state.current_page = 0

# # 启动的按钮
# if 'clicked' not in st.session_state:
#     st.session_state.clicked = False

# def click_button():
#     st.session_state.clicked = True

# st.button("Start Extracting", on_click = click_button)


# # if 'display' not in st.session_state:
#     # st.session_state.display = "### 所有控制行为规则： \n\n "

# if st.session_state.clicked:
#     # 开始时间
#     st.session_state.start_time = datetime.now()
#     # 结果文件名
#     unique_filename = f"result_{uuid.uuid4().hex}.md"
#     st.session_state.result_file = f"result/{unique_filename}"
#     my_bar = st.progress(0, text="Progress of Extraction")

#     progress = st.session_state.length * 3
#     current = 0

#     with st.chat_message("assistant"):
#         device_placeholder = st.empty()
#         topology_placeholder = st.empty()
#         rule_placeholder = st.empty()
#         results_placeholder = st.empty()


#         full_response = ""

#     for i in range(st.session_state.length):

#         retrieved_data = retrieve(st.session_state.raw_text[i], st.session_state.retriever)
#         formatted_prompt = chat_prompt_1.format(
#             context=retrieved_data["context"],
#             question=retrieved_data["question"]
#         )
#         response = st.session_state.llm_pipeline.run(formatted_prompt)
#         # 提取出的单点设备
#         temp_result_1 = response['response']['result'].content
#         device_placeholder.markdown(f"\n\n**piece{i+1}单点设备:**\n"+temp_result_1)
#         current += 1
#         my_bar.progress(current/progress, text = f"Progress of Extraction:{current/progress*100:.2f}%")

#         # 提取业务拓扑, 输入：工控系统文本和单点设备
#         formatted_prompt = chat_prompt_2.format(
#             text=st.session_state.raw_text[i],
#             device=temp_result_1
#         )
#         response = st.session_state.llm_pipeline.run(formatted_prompt)
#         temp_result_2 = response['response']['result'].content
#         topology_placeholder.markdown(f"\n\n**piece{i+1}业务拓扑:**\n"+temp_result_2)
#         current += 1
#         my_bar.progress(current/progress, text = f"Progress of Extraction:{current/progress*100:.2f}%")

#         # 提取业务逻辑，输入：工控系统文本和业务拓扑
#         formatted_prompt = chat_prompt_3.format(
#             text=result[i],
#             action=temp_result_2
#         )
#         response = st.session_state.llm_pipeline.run(formatted_prompt)
#         final_result = response['response']['result'].content
#         rule_placeholder.markdown(f"\n\n**piece{i+1}控制行为逻辑:**\n"+final_result)
#         current += 1
#         my_bar.progress(current/progress, text = f"Progress of Extraction:{current/progress*100:.2f}%")
#         # 立即写入文件
#         with open(f"result/{unique_filename}", "a", encoding="utf-8") as file:
#             file.write(str(final_result) + "\n")
#         # st.session_state.results.append(final_result) 
#         # st.session_state.display += final_result
#         # results_placeholder.markdown(st.session_state.display)

    
#     # 结束时间
#     st.session_state.end_time = datetime.now()

#     # 要插入的数据
#     data = {
#         "start_time":st.session_state.start_time,
#         "end_time":st.session_state.end_time,
#         "source_specification":st.session_state.data_value,
#         "pieces":st.session_state.length,
#         "LLM_parameters":{
#             "name":"gpt-4o-mini",
#             "temperature":st.session_state.temperature,
#             "top_p":st.session_state.top_p
#         },
#         "Knowledgebase_parameters":{
#             "knowledgebase_name":st.session_state.knowledgebase,
#             "semantic_based":st.session_state.semantic_based,
#             "keyword_based":st.session_state.keyword_based,
#             "embedding_model_name":"nomic-embed-text"

#         },
#         "result_file":st.session_state.result_file
#     }

#     chats_collection.insert_one(data)
#     st.success(f"Extraction has been finished and result is saved to {st.session_state.result_file}")

#     # 取消点击状态
#     st.session_state.clicked = False
#     # 消除文件相关内容的缓存
#     st.session_state["session_folder"] = ''
#     st.session_state["data_value"] = ''
#     st.session_state["raw_text"] = ''
#     st.session_state["length"] = ''



