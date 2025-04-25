# import streamlit as st
# import pandas as pd
# import numpy as np

# df = pd.DataFrame({
#     'first column':[1,2,3,4],
#     'second column':[10,20,30,40]
# })

# df

# x = 10

# 'x', '=',x

# dataframe= pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' %i for i in range(20))
# )

# st.table(dataframe)
# dataframe


# chart_data = pd.DataFrame(
#     np.random.randn(20,3),
#     columns=['a','b','c']
# )

# chart_data

# st.line_chart(chart_data)

# map_data = pd.DataFrame(
#     np.random.randn(1000,2)/[50,50]+[37.76, -122.4],
#     columns = ['lat', 'lon']
# )

# map_data

# st.map(map_data)


# y = st.slider('y',key = "lala")
# st.write(y, 'squared is', y*y)

# st.text_input("Your name", key="name")

# # st.session_state.name

# # st.session_state

# # st.write(st.session_state)



# def form_callback():
#     st.write(st.session_state.my_slider)
#     st.write(st.session_state.my_checkbox)

# with st.form(key = 'my_form'):
#     slider_input =st.slider('My slider', 0, 10, 5, key='my_slider')
#     checkbox_input = st.checkbox('Yes or No', key = 'my_checkbox')
#     submit_button = st.form_submit_button(label='Submit', on_click=form_callback)





# if st.checkbox('Show dataframe'):
#     chart_data = pd.DataFrame(
#         np.random.randn(20,3),
#         columns = ['a','b','c']
#     )

#     chart_data


# option = st.selectbox(
#     'Which number do you choose?',
#     df['first column']
# )

# 'You sekected:', option


# add_selectbox = st.sidebar.selectbox(
#     'How would you like to contacted?',
#     ('Email', 'Home phone', 'Mobile phone')
# )

# add_slider = st.sidebar.slider(
#     'Select a range of values',
#     0.0, 100.0, (25.0, 75.0)
# )

# left_column, right_column = st.columns(2)

# left_column.button('Press me!')

# with right_column:
#     chosen = st.radio(
#         'Sorting hat',
#         ('Gryffindor','Ravenclaw','Hufflepuff','Slytherin')
#     )
#     st.write(f"You are in {chosen} house!")


# import time 

# 'Starting a long computation...'

# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#     latest_iteration.text(f'Iteration {i+1}')
#     bar.progress(i+1)
#     time.sleep(0.1)

# '...and now we\'re done!'




# import streamlit as st
# import time

# with st.empty():
#     for seconds in range(10):
#         st.write(f"{seconds} seconds have passed")
#         time.sleep(1)
#     st.write(":material/check: 10 seconds over!")
# st.button("Rerun")




# import streamlit as st

# length = 50

# for i in range(length):
#     with st.tabs([i]):
#         st.header(f"{i}")


# import streamlit as st

# length = 50

# # 创建标签的列表
# tabs = [str(i) for i in range(length)]

# # 使用 st.tabs() 创建所有的标签
# selected_tab = st.tabs(tabs)

# # 使用 st.tabs() 作为上下文管理器来显示不同标签的内容
# for i, tab in enumerate(tabs):
#     with st.tabs([tab]):  # 使用每个标签名称
#         st.header(f"Tab {tab}")
#         st.write(f"Content for tab {tab}")


# import streamlit as st

# # 模拟分页数据
# rules = [
#     "规则 1: 这是第一页内容。",
#     "规则 2: 这是第二页内容。",
#     "规则 3: 这是第三页内容。",
#     "规则 4: 这是第四页内容。",
# ]

# # 页面索引
# page_index = st.session_state.get('page_index', 0)

# # 显示当前页的规则
# rule_placeholder = st.empty()
# rule_placeholder.markdown(rules[page_index])

# # 分页按钮
# col1, col2 = st.columns(2)

# with col1:
#     if page_index > 0:
#         if st.button('上一页'):
#             st.session_state.page_index = page_index - 1

# with col2:
#     if page_index < len(rules) - 1:
#         if st.button('下一页'):
#             st.session_state.page_index = page_index + 1


# my_tuple = (1, 2, 3, 4)
# print(my_tuple)

# single_element_tuple = (5,)
# print(single_element_tuple)

# another_tuple = 1, 2, 3
# print(another_tuple)

# print(my_tuple[0])
# print(my_tuple[-1])

# my_tuple = (1,2,3,4,5)

# sub_tuple = my_tuple[1:4]
# print(sub_tuple)

# last_two = my_tuple[-2:]
# print(last_two)

# my_tuple = (1,2,3,2,4,2)
# # 计算2出现的次数
# print(my_tuple.count(2))
# # 获取2的第一次出现的索引位置
# print(my_tuple.index(2))

# # 元组可以包含其他元组
# nested_tuple = ((1,2), (3,4), (5,6))

# # 访问嵌套元组的元素
# print(nested_tuple[0])
# print(nested_tuple[0][1])

# # 元组（tuple） 不可变：(1,2,3) ，用于存储不需要修改的数据，更快的操作，可以作为字典的键
# # 列表 (list) 可变：[1,2,3]，用于存储可能需要修改的数据，更慢的操作，不可以作为字典的键

# def get_coordinates():
#     return (10.0, 20.0)

# coordinates = get_coordinates()
# print(coordinates)


# def print_func(**kwargs):
#     print(type(kwargs))
#     print(kwargs)

# print_func(a=1, b=2, c='呵呵哒', d=[])


# def greet_me(**kwargs):
#     for key, value in kwargs.items():
#         print("{0} == {1}".format(key, value))

# greet_me(name="yasoob")


# from pymongo import MongoClient
# # 直接写死 MongoDB URI
# mongo_uri = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# # 连接到 MongoDB
# mongo_client = MongoClient(mongo_uri)

# # 访问数据库和集合
# db = mongo_client["rag"]  # 使用数据库 rag
# chats_collection = db["history"]  # 使用集合 history


# from pymongo import MnogoClient

# mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# mongo_client = MnogoClient(mongo_url)

# # 使用数据库 rag
# db = mongo_client["rag"]

# # 使用集合 history
# chats_collection = db["history"]

# 存储记录
# 开始的日期时间，结束的日期时间，原始文件的文件名（地址），【可供下载】
# 分成了多少piece，大模型相关参数，知识库相关参数，生成的结果文件名（地址）【可供下载】


# structure
# {start_time, end_time, source_specification, pieces, LLM_parameters, Knowledgebase_parameters, result_file}


# from datetime import datetime

# print(datetime.now())

# import uuid
# import os

# unique_filename = f"result_{uuid.uuid4().hex}.md"

# # 确保文件目录存在，如果不存在则创建
# if not os.path.exists("result"):
#     os.makedirs("result")

# with open(f"result/{unique_filename}", "a", encoding="utf-8") as file:
#     file.write("str(final_result)" + "\n")

# print(f"Result written to file: {unique_filename}")


# from pymongo import MongoClient
# from datetime import datetime
# # 直接写死 MongoDB URI
# mongo_url = "mongodb://localhost:27017/rag?retryWrites=true&w=majority"

# # 连接到 MongoDB
# mongo_client = MongoClient(mongo_url)

# # 访问数据库和集合
# db = mongo_client["rag"]  # 使用数据库 rag
# knowledgebase_collection = db["knowledgebase"]  # 使用集合 knowledgebase

# # knowledgebase 属性
# # collection_name(知识库集合名称):thermal_power  ; description(知识库描述):用于做什么 ; source(知识库来源): {filename:   ,chunk_size:  ,chunk_overlap:   }/ 人工搜集&专家提供;
# # create_time:datetime.now();   update_time:datetime.now();
# # embedding_model_name : nomic-embed-text

# data = {
#     "collection_name":"thermal_power",
#     "description": "火电厂操作员培训系统（OTS）仿真平台电气部分相关知识",
#     "source":"相关设备参数文档和专家提供的相关知识" ,
#     "create_time":datetime.now(),
#     "update_time":datetime.now(),
#     "embedding_model_name":"nomic-embed-text"
# }

# knowledgebase_collection.insert_one(data)


# quickStart.py
# import streamlit as st

# docs_file = r"C:\Users\ROOT\Desktop\project\RAG\result\result_d257a8f51e104b7496ee2495043e587f.md"

# def read_markdown_file(filepath):
#     with open(filepath, 'r', encoding='utf-8') as f:
#         return f.read()

# text = read_markdown_file(docs_file)
# st.markdown(text, unsafe_allow_html=True)





