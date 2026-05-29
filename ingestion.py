from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader # document loader
from langchain_text_splitters import CharacterTextSplitter # text splitter
from langchain_anthropic import ChatAnthropic # llm
from langchain_ollama import OllamaEmbeddings # embedding
from langchain_pinecone import PineconeVectorStore # vector store
# from pinecone import Pinecone
import os


if __name__ == "__main__":
    load_dotenv()

    docPat = "mediumblog1.txt"

    print("Loading document...")
    loader = TextLoader(docPat, encoding="utf-8")
    documents = loader.load()
    print("   Document loaded: ", len(documents), "pages with ", len(documents[0].page_content), "characters")

    print("Splitting document...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    texts = text_splitter.split_documents(documents)
    print("   Document splitted: ", len(texts), "chunks")

    print("Creating embedding...")
    OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL")
    embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    print("   Embedding created")

    print("Creating vector store...")
    INDEX_NAME = os.getenv("PINECONE_INDEX")
    vectorStore = PineconeVectorStore(embedding=embedding, index_name=INDEX_NAME)
    vectorStore.add_documents(texts)
    print("   Vector store created")