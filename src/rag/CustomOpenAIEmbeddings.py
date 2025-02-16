from openai import OpenAI

class CustomOpenAIEmbeddings:
    def __init__(self, model="text-embedding-ada-002", openai_api_key=None, base_url="https://api.openai-proxy.org/v1"):
        """
        初始化自定义的嵌入类。
        :param model: 要使用的 OpenAI 嵌入模型 (默认为 "text-embedding-ada-002")
        :param openai_api_key: OpenAI API 密钥
        :param base_url: 可选的 OpenAI API 代理 URL
        """
        self.model = model
        self.api_key = openai_api_key
        self.base_url = base_url

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def embed_query(self, text):
        """
        获取单个文本的嵌入向量。
        :param text: 输入文本
        :return: 嵌入向量 (列表)
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_documents(self, texts):
        """
        获取多个文本的嵌入向量。
        :param texts: 输入文本列表
        :return: 嵌入向量列表
        """
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
