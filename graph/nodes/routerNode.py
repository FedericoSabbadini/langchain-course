from typing import Literal, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from graph.states import GraphState
from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class routerNodeClass:
    def __init__(self):
        llm = ChatOpenAI(temperature=0)
        structured_llm_router = llm.with_structured_output(RouteQuery)

        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. For all else, use web-search."""
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )
        self.chain = route_prompt | structured_llm_router
        self.name = "route_question"

    def __call__(self, state: GraphState) -> Dict[str, str]:
        print("---ROUTE QUESTION---")
        question = state["question"]
        source: RouteQuery = self.chain.invoke({"question": question})
        return {"datasource": source.datasource}