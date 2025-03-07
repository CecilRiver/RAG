
from src.rag.vector_store import VectorStoreManager
from src.rag.text_splitter import TextSplitter
import os
from langchain.docstore.document import Document


# 每个文档至少 50-100 字，提高词频和信息量


# 构造知识库文档
docs = [
    Document(
        page_content="世界上最大的大陆是亚洲，亚洲的总面积约为 4457 万平方千米，占全球陆地总面积的 30%。"
                     "亚洲拥有世界上最多的人口，总人口超过 46 亿。亚洲的地形复杂，包括高山、平原、盆地等，"
                     "其中珠穆朗玛峰是全球最高的山峰。亚洲还涵盖多个国家，包括中国、印度、俄罗斯等。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="亚洲不仅是全球最大的大陆，同时也是文化最丰富的地区之一。亚洲大陆的地势东高西低，"
                     "拥有世界上最广阔的平原之一——西西伯利亚平原，以及世界上最深的湖泊——贝加尔湖。"
                     "此外，亚洲拥有多个世界文明古国，如中国、印度和巴比伦。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="北美洲是地球上的第三大洲，总面积约 2470 万平方千米。北美洲包括美国、加拿大、墨西哥等国家。"
                     "北美洲的地形多样，既有落基山脉这样的高山，也有密西西比河这样的世界级大河流。"
                     "此外，北美洲拥有世界上最大的淡水湖群——五大湖。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="南极洲是地球上最冷的大陆，其面积约 1400 万平方千米，几乎全部被厚厚的冰层覆盖。"
                     "南极洲的气候极端寒冷，年均气温低于 -50°C，是全球最干燥的地区之一。"
                     "由于其极端环境，南极洲没有固定居民，主要由科学家在科研站进行考察和研究。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="撒哈拉沙漠是世界上最大的沙漠，位于非洲北部，面积高达 920 万平方千米。"
                     "撒哈拉沙漠气候极端干燥，年降水量通常不足 50 毫米。"
                     "然而，在其极端气候下，仍然生存着许多耐旱植物和动物，如仙人掌、骆驼和沙狐。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="世界最高的山峰是珠穆朗玛峰，位于亚洲的喜马拉雅山脉，海拔 8848.86 米。"
                     "珠峰是登山者的终极挑战，每年有许多登山者尝试攀登，但由于高海拔、极端气候和缺氧环境，"
                     "登顶成功率并不高。喜马拉雅山脉是印度板块与欧亚板块碰撞形成的。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="地球的总面积约为 5.1 亿平方千米，其中 71% 被海洋覆盖，仅 29% 为陆地。"
                     "地球的水资源主要分布在太平洋、大西洋和印度洋，这三大洋共同影响全球气候。"
                     "地球的生态系统包括森林、草原、湿地、沙漠等，维持着生物多样性。",
        metadata={"category": "地球科学"}
    ),
    Document(
        page_content="长江是中国最长的河流，全长 6300 公里，发源于青藏高原的唐古拉山脉，"
                     "流经中国多个省份，最终汇入东海。长江是中国最重要的水源之一，"
                     "也是全球水运最繁忙的河流之一，沿岸经济发达。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="尼罗河是世界上最长的河流，全长约 6650 公里，流经非洲 11 个国家，最终注入地中海。"
                     "尼罗河被称为“埃及的母亲”，因为它为埃及及其周边地区提供了农业灌溉和饮用水。"
                     "古埃及文明的兴起与尼罗河的周期性泛滥密不可分。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="太平洋是世界上最大的海洋，面积约 1.65 亿平方千米，占全球海洋总面积的 50%。"
                     "太平洋横跨亚洲、美洲和大洋洲，拥有马里亚纳海沟——全球最深的海沟，"
                     "深度超过 1 万米。它是全球气候变化的重要影响因素。",
        metadata={"category": "海洋学"}
    ),
    Document(
        page_content="中国的官方语言是普通话，而英语是世界上最广泛使用的语言，"
                     "作为第一语言和第二语言使用的人口超过 15 亿。"
                     "西班牙语和法语也是全球重要的语言，在多个国家被广泛使用。",
        metadata={"category": "语言"}
    ),
    Document(
        page_content="世界上人口最多的国家是中国，总人口超过 14 亿。"
                     "印度是世界第二人口大国，总人口接近 14 亿。"
                     "全球人口分布不均，亚洲是世界上人口最稠密的地区，而南极洲几乎无人居住。",
        metadata={"category": "人口"}
    ),
    Document(
        page_content="俄罗斯是世界上面积最大的国家，国土总面积 1709 万平方千米，横跨欧亚两大洲。"
                     "俄罗斯的地理多样性极高，既有西伯利亚的寒冷冻土，也有黑海沿岸的温带气候区。",
        metadata={"category": "地理"}
    ),
    Document(
        page_content="非洲的刚果盆地是地球第二大雨林区，仅次于南美洲的亚马逊雨林。"
                     "刚果盆地覆盖多个非洲国家，提供丰富的生物多样性，对全球气候调节起到关键作用。",
        metadata={"category": "生态"}
    ),
    Document(
        page_content="现代计算机的基础理论由英国数学家艾伦·图灵奠定，他被誉为计算机科学之父。"
                     "图灵机概念奠定了计算理论的核心基础，为人工智能和计算机科学的发展做出了巨大贡献。",
        metadata={"category": "计算机科学"}
    ),
    Document(
        page_content="第一颗人造卫星是苏联于 1957 年 10 月 4 日发射的 Sputnik 1 号。"
                     "它标志着人类正式进入太空时代，拉开了美苏太空竞赛的序幕。"
                     "Sputnik 1 仅 58 厘米宽，重 83.6 公斤，环绕地球运行了 21 天。",
        metadata={"category": "科技史"}
    ),
    Document(
        page_content="黑洞是一种引力极强的天体，连光都无法逃逸。"
                     "黑洞的质量极大，密度极高，任何物质进入其视界后都无法逃脱。"
                     "科学家通过观察黑洞周围的物质运动，间接推测黑洞的存在。",
        metadata={"category": "天文学"}
    ),
    Document(
        page_content="相对论由著名物理学家阿尔伯特·爱因斯坦提出，分为狭义相对论和广义相对论。"
                     "相对论彻底改变了人类对时间和空间的理解，并且为现代 GPS 卫星导航提供了理论基础。",
        metadata={"category": "物理"}
    ),
    Document(
        page_content="太阳系包含 8 颗行星，包括水星、金星、地球、火星、木星、土星、天王星和海王星。"
                     "地球是唯一已知存在生命的星球，位于宜居带。",
        metadata={"category": "天文学"}
    ),
    Document(
        page_content="第二次世界大战的重要转折点之一是 1944 年的诺曼底登陆，"
                     "盟军在法国海岸成功开辟欧洲战场，加速了纳粹德国的失败。",
        metadata={"category": "历史"}
    ),
]


# 初始化Chroma向量数据库
vector_store_manager = VectorStoreManager(
            vector_store_type="chroma",
            collection_name="langchain_collection",
            embedding_model_name = "nomic-embed-text",
            #embedding_model_name="llama3",
            embedding_type="llama"
        )

#添加文档到Chroma
vector_store_manager.add_documents(docs)


# 从Chroma 取出所有数据，并转化为Document 格式
chroma_docs = vector_store_manager.vector_store.get()
bm25_docs = [
    Document(page_content=text, metadata = metadata)
    for text, metadata in zip(chroma_docs["documents"],chroma_docs["metadatas"])
]

print(bm25_docs)

# # 向量存储
# vector_store_manager = VectorStoreManager(
#             vector_store_type="chroma",
#             collection_name="langchain_collection"
#         )

# # 知识库清空
# vector_store_manager.clear_vector_store()


# # 文本分割
# splitter = TextSplitter(
#     splitter_type = "recursive",
#     chunk_size = 100,
#     chunk_overlap = 20
# )


# # 知识文件地址
# data = r"C:\Users\ROOT\Desktop\示例.pdf"

# if(os.path.isfile(data)):
#     docs = splitter.split_file_documents([data])
# else:
#     raise ValueError("Invalid data_value for 'file' data_task. Must be a file path or list of file paths.")

# # 分割后文本加入知识库中
# vector_store_manager.add_documents(docs, clear_store=True)


# # 检索示例
# retriever = vector_store_manager.as_retriever()

# print(retriever.get_relevant_documents("推理步骤"))