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
    model="llama3"
)

input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])