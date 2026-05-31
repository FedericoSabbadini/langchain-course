from langgraph.graph import MessagesState
from .toolNode import tools
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
load_dotenv()


SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

model = os.getenv("OLLAMA_MODEL", "gpt-4o-mini")
llm = ChatOllama(model=model).bind_tools(tools)

def agentReason_node(state: MessagesState) -> MessagesState:
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )

    return {"messages": [response]}