from typing import Any, Dict
from langchain_chroma import Chroma
from langchain_anthropic import OpenAIEmbeddings
from graph.states import GraphState
from ingestion import retriever


class retrieverNodeClass:
    def __init__(self) -> None:
        self.retriever = Chroma(
            collection_name="rag-chroma",
            persist_directory="./.chroma",
            embedding_function=OpenAIEmbeddings(),
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
