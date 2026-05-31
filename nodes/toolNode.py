from dotenv import load_dotenv
load_dotenv()
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )


def toolNodeClass(search_queries: list[str], **kwargs):
    """Run the generated queries. This function will be used as a tool in the chain, so it needs to have this signature.
     It will receive the output of the chain as input, so it needs to be able to handle the output of the chain, which is a dictionary with a key "search_queries" that contains a list of search queries.
    """

    return TavilySearch(max_results=5).batch([{"query": query} for query in search_queries])


toolNode = ToolNode(
    [
        StructuredTool.from_function(toolNodeClass, name=AnswerQuestion.__name__),
        StructuredTool.from_function(toolNodeClass, name=ReviseAnswer.__name__),
    ]
)