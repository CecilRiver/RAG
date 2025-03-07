from src.llm.llm_pipeline import LLMPipeline
from src.rag.vector_store import VectorStoreManager
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.docstore.document import Document

import jieba

# llama_pipeline = LLMPipeline(model_type="ollama", model_path="deepseek-r1", n_ctx=512, num_threads=4)

# llama_response = llama_pipeline.run("世界上最大的大陆是哪个?")
# print("Llama Response:")
# print(llama_response)


# 使用jieb进行分词
def chinese_tokenizer(text):
    return list(jieba.cut(text, cut_all=False))

# print("BM25 分词测试:")
# print(chinese_tokenizer("世界上最大的大陆是哪个？"))
# print(chinese_tokenizer("二战最重要的转折点之一是 1944 年的诺曼底登陆。"))


vector_store_manager = VectorStoreManager(
            vector_store_type="chroma",
            collection_name="langchain_collection",
            embedding_model_name = "nomic-embed-text",
            #embedding_model_name="llama3",
            embedding_type="llama"
        )

retriever = vector_store_manager.as_retriever()


# 从Chroma 取出所有数据，并转化为Document 格式
chroma_docs = vector_store_manager.vector_store.get()
bm25_docs = [
    Document(page_content=text, metadata = metadata)
    for text, metadata in zip(chroma_docs["documents"],chroma_docs["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(
    bm25_docs,
    tokenizer = chinese_tokenizer # 显式指定中文分词器
    )
bm25_retriever.k = 5

# 初始化集成检索器
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, retriever],weights=[0.5,0.5])


# 测试查询
query = "世界上最大的大陆是哪个？"



# BM25的检索
print("BM25的检索：")
bm25_results = bm25_retriever.invoke(query)
for idx, doc in enumerate(bm25_results):
    print(f"BM25 文档 {idx+1} : {doc.page_content}\n")


# Chroma向量检索
print("Chroma向量搜索：")
vector_results = retriever.get_relevant_documents(query)
for idx, doc in enumerate(vector_results):
    print(f"Chroma 文档 {idx+1}： {doc.page_content}\n")




results = ensemble_retriever.invoke(query)

print("混合 Ensemble 检索结果")
# 打印结果
for idx,doc in enumerate(results):
    print(f"文档{idx+1}：{doc.page_content}\n")


# query = "无人驾驶汽车的定义与分类是什么?"

# rag_response = llama_pipeline.generate_response(query, retriever)
# print("Retrieval-Augmented Generation Response:")
# print(rag_response)

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