from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# ---------------------------
# CONFIG
# ---------------------------
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# ---------------------------
# LOAD DOCS (ROBUST VERSION)
# ---------------------------
docs_list = []

for url in urls:
    try:
        print(f"Loading: {url}")

        loader = WebBaseLoader(
            url,
            requests_kwargs={
                "timeout": 10,
                "headers": {
                    "User-Agent": "Mozilla/5.0"
                }
            }
        )

        docs = loader.load()
        docs_list.extend(docs)

    except Exception as e:
        print(f"FAILED URL: {url}")
        print(e)

# ---------------------------
# SPLITTING
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=250
)

doc_splits = text_splitter.split_documents(docs_list)

print(f"Total chunks: {len(doc_splits)}")

# ---------------------------
# EMBEDDINGS + VECTORSTORE
# ---------------------------
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL")
    ),
    persist_directory="./.chroma",
)