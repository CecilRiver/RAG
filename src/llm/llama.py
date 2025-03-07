import os
from dotenv import load_dotenv
from textwrap import dedent
import json
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain_community.llms import Ollama
load_dotenv()

class LlamaChatbot:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        num_threads: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9
    ):
        """
        Preserves your original constructor signature, but now we’ll
        instantiate an OllamaLLM for multi-turn usage.
        
        :param model_path:    Path or name of your Llama/Ollama model.
        :param n_ctx:         Context window size.
        :param num_threads:   Threads to use for inference.
        :param temperature:   Sampling temperature.
        :param max_tokens:    Max tokens in the final answer.
        :param top_p:         Nucleus sampling.
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.num_threads = num_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # 1) 为链创建一个 OllamaLLM 实例
        #    （如果你想使用 LlamaCpp，请在这里替换。）
        self.llm = OllamaLLM(
            model=model_path,
            n_ctx=n_ctx,
            num_threads=num_threads,
            temperature=temperature,
            top_p=top_p,
            # 如果你的 LLM 支持，可以传递 max_tokens 参数，
            # 但并非所有的封装器都直接支持它。
        )

       # 2) 为多轮对话准备记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # 3) 创建完链之后，我们将存储它
        self.conversation_chain = None


    def create_prompt_template(self, template: str, input_variables: list[str]) -> PromptTemplate:
        """
        Returns a basic PromptTemplate.
        """
        return PromptTemplate(template=template, input_variables=input_variables)


    def create_retrieval_qa_chain(self, retriever, chain_type="stuff", prompt_template=None):
        """
        Original method built a RetrievalQA; now we build a ConversationalRetrievalChain
        for RAG. We'll store it in self.conversation_chain.

        :param retriever:       A retriever from your vector store.
        :param chain_type:      Unused, kept for signature compatibility.
        :param prompt_template: (Optional) If provided, a custom ChatPromptTemplate for system + user steps.
        :return:                The conversation chain.
        """
        # 如果没有提供自定义的提示，定义一个简单的系统提示，引用 {context}
        if prompt_template is None:
            system_template = """
            你是一名先进且知识渊博的 AI 助手，致力于为用户提供有帮助、准确且详细的回答。你可以访问并利用检索增强生成（RAG）提供的上下文信息，同时在上下文不足、无关或缺失时，也能够依靠自身的常识补充回答。

            {context}
            
            ## 关键指引：
            1. **上下文的使用：**
            - 只要提供的上下文与问题直接相关，就应当在回答中使用它。
            - 如果上下文信息部分缺失或不完整，请适当整合，并明确指出其局限性。
            - 如果上下文与用户的问题无关，应明确说明：“提供的上下文与该问题无关。”

            2. **处理不足或缺失的上下文：**
            - 如果没有提供上下文，或上下文未能回答用户问题：
                - 需要明确说明缺少相关上下文。
                - 使用你自己的通用知识回答问题，确保回答准确、全面，并适当控制回答范围。

            3. **平衡上下文信息与常识：**
            - 清晰区分基于上下文的信息和基于通用知识的见解。
            - 避免优先采用无关或冲突的上下文，而忽略更可信的常识。

            4. **透明性和解释性：**
            - 明确说明信息来源：
                - 对于基于上下文的信息，使用“根据提供的上下文...”。
                - 对于超出上下文范围的常识性信息，使用“在提供的上下文之外...”或“基于通用知识...”。
            - 必要时，清楚解释回答的推理逻辑或依据。

            5. **以用户为中心的适应性：**
            - 依据用户的需求调整回答，确保表达清晰。
            - 适当时提供后续建议（如“你可以通过...获取更多信息”）或进一步澄清疑问。

            ## 回答格式：
            - 直接回答用户的问题。
            - 如果使用了上下文，请说明相关信息的来源。
            - 如果上下文无关或缺失，请明确说明，并基于通用知识回答问题。
            - 适当时，提供分步骤推理或示例，以增强理解。

            ## 示例行为：
            1. **有相关上下文：**
            - 问题："法国的首都是哪里？"
            - 上下文："法国是欧洲的一个国家，首都是巴黎。"
            - 回答："根据提供的上下文，法国的首都是巴黎。"

            2. **上下文可用但无关：**
            - 问题："德国的首都是哪里？"
            - 上下文："法国是欧洲的一个国家，首都是巴黎。"
            - 回答："提供的上下文与该问题无关。根据通用知识，德国的首都是柏林。"

            3. **没有提供上下文：**
            - 问题："世界上最高的山是什么？"
            - 上下文：无。
            - 回答："没有提供相关上下文。根据通用知识，世界上最高的山是珠穆朗玛峰，高度为 8,848 米（29,029 英尺）。"

            4. **部分上下文可用：**
            - 问题："请介绍欧洲的可再生能源。"
            - 上下文："德国是太阳能应用的领先国家。"
            - 回答："根据提供的上下文，德国在太阳能应用方面处于领先地位。在提供的上下文之外，欧洲在风能和水电方面也处于全球领先地位，例如丹麦和挪威做出了重大贡献。"

            ## 额外规则：
            - 避免猜测。如果不确定，应说明：“我对此不太确定，但可以提供一般性的见解或建议相关资源。”
            - 力求完整回答，但也要尊重用户的问题和上下文范围。
            - 保持专业、友好和易于理解的语气。
            """
            
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        else:
            # 如果用户提供了 PromptTemplate，假设它是一个 ChatPromptTemplate 或者可以类似使用
            chat_prompt = prompt_template

        # 构建一个多轮检索链
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )
        return self.conversation_chain
    

    def chat(self, messages):
        chat_model = ChatOllama(model=self.model_path, n_ctx=self.n_ctx, num_threads=self.num_threads)
        response = chat_model.invoke(messages)
        return {"result": response}

    def generate_response(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Single-turn usage with conversation chain. We create or update the chain,
        then pass the query as {"question": ...}.

        :param query:           The user's question.
        :param retriever:       VectorStore retriever for doc context.
        :param prompt_template: (Optional) ChatPromptTemplate or similar.
        :param chain_type:      Unused, for signature compatibility.
        :return:                The final answer string.
        """
        
        if prompt_template is None:
            prompt = hub.pull("hwchase17/multi-query-retriever")
            
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template = prompt_template)
        result = self.conversation_chain({"question": query})
        return result["answer"]

    def generate_response_with_sources(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Same as above, but return the sources too.

        :param query:           The user's question.
        :param retriever:       VectorStore retriever.
        :param prompt_template: Optional ChatPromptTemplate.
        :param chain_type:      Unused, for signature compatibility.
        :return:                Dict with {"result": <answer>, "sources": <list of documents>}
        """
        if prompt_template is None:
            prompt = hub.pull("hwchase17/multi-query-retriever")
        

        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template = prompt_template)
        result = self.conversation_chain({"question": query})
        return {
            "result": result["answer"],
            "sources": result["source_documents"]
        }

# Example usage
if __name__ == "__main__":
    # model_path = "./model/zephyr-7b-beta.Q4_0.gguf"
    # Create the LlamaChatbot
    bot = LlamaChatbot(model_path="./models/your-llama-model.bin")

    # Create a prompt template
    template = """
    You are a helpful assistant that answers questions based on the provided context.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    prompt_template = bot.create_prompt_template(template, ["context", "query"])

    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = chatbot.chat(messages)
    print(response)

    # Example RAG usage
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OllamaEmbeddings

    # Suppose you have a retriever from Pinecone or Chroma
    retriever = my_vectorstore.as_retriever()

    # Single-turn usage
    answer_text = bot.generate_response("What is quantum entanglement?", retriever)
    print("Answer:", answer_text)

    # With sources
    res = bot.generate_response_with_sources("Tell me about black holes", retriever)
    print("Answer:", res["result"])
    print("Sources:", res["sources"])

    # Multi-turn usage
    bot.create_retrieval_qa_chain(retriever)
    messages = [{"role": "user", "content": "Who discovered penicillin?"}]
    resp = bot.chat(messages)
    print("Assistant:", resp)

    # Next user query referencing prior answer
    messages.append({"role": "user", "content": "What year was that?"})
    resp2 = bot.chat(messages)
    print("Assistant:", resp2)
