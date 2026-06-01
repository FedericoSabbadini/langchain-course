from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question, useful to keep tracked because it can be used to generate the next question
        generation: LLM generation, useful to keep tracked because it can be used to generate the next question
        web_search: whether to add search, useful to keep tracked because it can be used to generate the next question
        documents: list of documents, useful to keep tracked because it can be used to generate the next question
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    datasource: str

# Example of how to use the GraphState
# state = GraphState(
#     question="What is the capital of France?",
#     generation="The capital of France is Paris.",
#     web_search=False,
#     documents=["Document 1", "Document 2"],
# )
# The return of nodes will be in the form:
# {
#     "question": "What is the capital of France?",
#     "generation": "The capital of France is Paris.",
#     "web_search": False,
#     "documents": ["Document 1", "Document 2"],
# }