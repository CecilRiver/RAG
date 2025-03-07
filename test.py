# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# template = """Question: {question}

# Answer: Let's think step by step."""

# prompt = ChatPromptTemplate.from_template(template)

# model = OllamaLLM(model="deepseek-r1:latest")
# chain = prompt | model
# response = chain.invoke({"question": "What is LangChain?"})
# print(response)



# from openai import OpenAI

# client = OpenAI(
#     base_url='https://api.openai-proxy.org/v1',
#     api_key='sk-IAJU4HVFoab6msNQIgCW1lU716arBT7LW8iCTFAwRm9lqB7t',
# )


# response = client.embeddings.create(
#     input="Your text string goes here",
#     model="text-embedding-ada-002"
# )

# print(response.data[0].embedding)


from langchain_ollama import OllamaEmbeddings

embed = OllamaEmbeddings(
    model="nomic-embed-text"
)

input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])


# name = 'Lemon'
# age = 18

# print(f"my name is {name}, age is {age}")

# n1 = 3.1415926
# print(f"保留两位小数：{n1:.2f}")

# s1 = "test1"
# s2 = "test2"
# print(s1+s2)
# print(s1*2)
# print(s1,s2)
# print(s1.title())
# print(s1.upper())
# print(s1+"\n"+s2)

# s = ' hello, welcome to PyDataLab  '
# print(s.split('e',1))
# print(s)
# print(s.strip())
# print(s.encode(encoding='utf-8'))
# print(s.count('e'))
# print(len(s))
# print(len(s.strip()))


# names = ['James','Michael','Emma','Emily']
# print("names的数据类型是",type(names))
# print(names)
# print(len(names))
# print(names[-1])
# print(names[-2])
# for name in names:
#     print(name)
# for i in range(len(names)):
#     print(names[i])
# names.append('Jacob')
# names.insert(1,"ABB")
# for name in names:
#     print(name)
# names.pop()
# print(names.pop(0))
# for i in range(len(names)):
#     print(names[i])

# print("列表相加：",[1,2,3]+['a','b'])
# print("列表相乘：",['a','b']*3)
# print("判断元素是否存在于列表中：",'a' in ['a','b'])
# print("判断元素是否存在于列表中：",'a'  not in ['a','b'])


# a_list = ['Lemon', 100, ['a','b','c','d'],True]

# # new_list[start: end : step],选取元素包含start，不包含end
# c_list = [1,2,3,4,5,6]
# print(c_list[1:3])
# print(c_list[::2])
# print(c_list[0:len(c_list)-1:2])
# print(c_list[::-1])
# print(c_list[::-2])

# # 列表推导式
# str_list = [x.lower() for x in "Lemon"]
# print(str_list)
# list_list = [x**2 for x in [1,2,3,4]]
# print(list_list)
# tuple_list = [x+2 for x in (1,2,4,4)]
# print(tuple_list)
# d = {'x':'1','y':'2','z':'4'}
# d_list = [k+'='+v for k,v in d.items()]
# print([d_list[i] for i in [0,2]])
# print(d_list)
# if_list = [i**2 for i in range(10) if i%2==0]
# print(if_list)
# print([i**2 if i%2 == 0 else i+2 for i in range(10)])
# print([i**2 for i in range(10) if i%2==0 if i%3==0])


# def get_keys(d, value):
#     return [k for k,v in d.items() if v ==value]

# my_dict = {'name':'John','age':25,1:[2,4,3]}
# print("copy:",my_dict.copy())
# print("keys",my_dict.keys())
# print('values',my_dict.values())
# print('items',my_dict.items())
# my_dict.pop('age')
# print(my_dict)
# print(get_keys(my_dict,'John'))
# my_dict.popitem()
# print(my_dict)

# my_dict01 = {x:x*x for x in range(6)}
# print(my_dict01)
# my_list = [x*x for x in range(6)]
# print(my_list)



# import datetime
# # 日期
# print(datetime.date.today())

# #日期＋时间
# print(datetime.datetime.now())

# import time
# print(time.time())
# print(time.localtime())
# print(time.gmtime())


# import calendar
# calendar.prcal(2025)




# import numpy as np

# # 创建ndarray数组
# # 基于list
# arr1 = np.array([1,2,3,4])
# print(arr1)
# # 基于tuple
# arr_tuple = np.array((1,2,3,4))
# print(arr_tuple)

# arr2 = np.array([[1,2,4],[3,4,5]])
# print(arr2)


# # 基于np.arange
# arr3 = np.arange(5)
# print(arr3)

# arr4 = np.array([np.arange(3),np.arange(3)])
# print(arr4)

# arr = np.arange(24).reshape(2,3,4)
# print(arr)

# print(np.arange(5, dtype=float))

# a=np.array([[1,2,3],[7,8,9]])
# print(a.ndim)
# print(a.shape)
# print(a.size)
# print(a.itemsize)

# b = np.arange(24).reshape(4,6)
# print(b)
# print(b.T)

# for item in b.T.flat:
#     print(item)
# print(b[0:3,0:2])

# print(np.random.rand(4,3,2))

# #具有标准正态分布
# print(np.random.randn(4,3,2))

# print(np.random.choice(3,3))

# print(np.random.choice(3,3,replace=False))

# np.random.seed(1676)
# print(np.random.rand(6))
# np.random.seed(1676)
# print(np.random.rand(6))


# from src.llm.llm_pipeline import LLMPipeline
# from src.rag.vector_store import VectorStoreManager
# from src.rag.text_splitter import TextSplitter

# import os
# # Llama pipeline
# llama_pipeline = LLMPipeline(model_type="ollama", model_path="deepseek-r1", n_ctx=512, num_threads=4)
# llama_response = llama_pipeline.run("世界上最大的大陆是哪个?")
# print("Llama Response:")
# print(llama_response)

# # Chat example
# messages = [
#     {"user": "嗨，你好吗？", "assistant": "我很好，谢谢！今天有什么可以帮助你的吗？"},
#     {"user": "你能告诉我一个笑话吗？", "assistant": "当然！给你讲个笑话：为什么科学家不相信原子？因为它们什么都能编造出来！"},
#     {"user": "这个不错！你还知道其他科学笑话吗？", "assistant": ""}
# ]
# chat_response = llama_pipeline.chat(messages)
# print("Chat Response:")
# print(chat_response)

# Retrieval-augmented generation example

# vector_store_manager = VectorStoreManager(
#             vector_store_type="chroma",
#             collection_name="langchain_collection"
#         )
# print('---------------------------------------------------------------------------------')
# vector_store_manager.clear_vector_store()
# result = vector_store_manager.vector_store.get()
# print(result)
# with open("vector_store_output.txt","w",encoding="utf-8") as file:
#     file.write(str(result))
# print(dir(vector_store_manager.vector_store))
# print(vector_store_manager.vector_store.get_by_ids('bd7fadaa-8ac8-4c14-bded-c3fd4c37ce26'))

# data = vector_store_manager.vector_store.get()  # 获取所有存储数据
# target_id = "bd7fadaa-8ac8-4c14-bded-c3fd4c37ce26"

# if 'ids' in data and 'documents' in data:
#     try:
#         index = data['ids'].index(target_id)
#         print("Found document:", data['documents'][index])
#     except ValueError:
#         print("ID not found")



# splitter = TextSplitter(
#     splitter_type = "recursive",
#     chunk_size = 100,
#     chunk_overlap = 20
# )

# data = r"C:\Users\ROOT\Desktop\示例.pdf"

# if(os.path.isfile(data)):
#     docs = splitter.split_file_documents([data])
# else:
#     raise ValueError("Invalid data_value for 'file' data_task. Must be a file path or list of file paths.")

# print(docs)

# vector_store_manager.add_documents(docs, clear_store=True)

# retriever = vector_store_manager.as_retriever()

# print(retriever.get_relevant_documents("推理步骤"))

# retriever = vector_store_manager.as_retriever()
# docs = retriever.get_relevant_documents("凝固系数")
# print(docs)
# print('---------------------------------------------------------------------------------')
# # Assuming you have a Chroma vector store named 'vector_store'
# retriever = vector_store_manager.as_retriever()

# query = "无人驾驶汽车的定义与分类是什么?"
# # rag_response = llama_pipeline.generate_response(query, retriever)
# # print("Retrieval-Augmented Generation Response:")
# # print(rag_response)

# rag_response_with_sources = llama_pipeline.generate_response_with_sources(query, retriever)
# print("Retrieval-Augmented Generation Response with Sources:")
# print(rag_response_with_sources["result"])
# print("Sources:")
# for source in rag_response_with_sources["sources"]:
#     print('---------------------------------------------------------------------------------')
#     print(source)
#     print('---------------------------------------------------------------------------------')
#     print(source.metadata["source"])
#     print('---------------------------------------------------------------------------------')





from langchain_community.retrievers import BM25Retriever
print(BM25Retriever)
