from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings # embedding
from langchain_anthropic import ChatAnthropic # llm
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore # vector store
from langchain_core.prompts import PromptTemplate # prompt template
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser   
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# RunnablePassthrough: A runnable that simply passes its input through 
# without modification. It can be used to chain together other
# runnables while preserving the original input.
from operator import itemgetter
import os

load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX")
KEY_PINECONE = os.getenv("PINECONE_API_KEY")
EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

embedder= OllamaEmbeddings(model=EMBEDDING_MODEL)
# llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL"), temperature=0.9, anthropic_api_key="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")
llm = init_chat_model(model="ollama: " + OLLAMA_MODEL)
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embedder
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context, 
    if the question cannot be answered based on the context, say  "I don't know". 
    Do not use any information that is not in the context.:

{context}

Question: {question}

Provide a detailed answer:"""
)
def format_docs_as_retrieved_info(docs):
    retrieved_info = ""
    for i, doc in enumerate(docs):
        retrieved_info += f"Document {i+1}:\n{doc.page_content}\n\n"
    return retrieved_info
    


query = "Tell me about Pinecone in machine learning."
print(f"Query: {query}")

# print('----Answering without retrieved information:----')
# prompt = prompt_template.format_messages(
#     context="",
#     question=query
# )
# response = llm.invoke(prompt)
# print(response.content)

# print('\n----Answering with retrieved information:----')
# retrieved_docs = retriever.invoke(query)
# print(f"Retrieved {len(retrieved_docs)} documents.")
# retrieved_info = format_docs_as_retrieved_info(retrieved_docs)
# prompt = prompt_template.format_messages(
#     context=retrieved_info,
#     question=query
# )
# response = llm.invoke(prompt)
# print(response.content)

print('\n----Answering with retrieved information with chain:----')
question = {"question": query}
chain = (
    RunnablePassthrough.assign( # .assign allows us to create a new runnable that takes the output of the previous runnable and assigns it to a new variable. 
        # In this case, we are taking the "question" from the input and assigning it to a new variable called "context" that will be used in the next step of the chain.
        context=itemgetter("question")
        | retriever # with RunnablePassthrough, the only input to tetriever is a string, not a dictionary
        | format_docs_as_retrieved_info # the method output is put in the "context" variable, which is used in the next step of the chain
    ) # question -> (question, context)
    | prompt_template # (question, context) -> prompt
    | llm # prompt -> response
    | StrOutputParser() # response -> output
)
response = chain.invoke(question)
print(response)
