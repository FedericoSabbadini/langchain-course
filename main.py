from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings # embedding
from langchain_anthropic import ChatAnthropic # llm
from langchain_pinecone import PineconeVectorStore # vector store
from langchain_core.prompts import PromptTemplate # prompt template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import os

load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX")
KEY_PINECONE = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")

embedder= OllamaEmbeddings(model=EMBEDDING_MODEL)
llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL"), temperature=0.9, anthropic_api_key="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")
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedder
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """
        You are a helpful assistant for answering questions about the content of a document.
        Use the following pieces of retrieved information to answer the question at the end. 
        If you don't know the answer, say you don't know.
        {retrieved_info}
        Question: {question}
        Answer:
    """)

def format_docs_as_retrieved_info(docs):
    retrieved_info = ""
    for i, doc in enumerate(docs):
        retrieved_info += f"Document {i+1}:\n{doc.page_content}\n\n"
    return retrieved_info
    
query = "What is Pinecone in machine learning?"
print(f"Query: {query}")
print('----Answering without retrieved information:----')
prompt = prompt_template.format_prompt(
    retrieved_info="",
    question=query
).to_messages()
response = llm.invoke(prompt)
print(response.content)
