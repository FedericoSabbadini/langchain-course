from typing import Any, Dict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings # embedding
from graph.states import GraphState
import os

class retrieverNodeClass:
    def __init__(self) -> None:
        self.retriever = Chroma(
            collection_name="rag-chroma",
            persist_directory="./.chroma",
            embedding_function=OllamaEmbeddings(model=os.getenv("OLLAMA_EMBEDDING_MODEL")),
        ).as_retriever()
        self.name = "retrieve"


    def __call__(self, state: GraphState) -> Dict[str, Any]:
        print("---RETRIEVE---")
        question = state["question"]

        documents = self.retriever.invoke(question)
        return {
            "documents": documents, 
            "question": question
            }
