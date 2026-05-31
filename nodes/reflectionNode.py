from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from state import MessageGraph
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()


class reflectionNodeClass:
    def __init__(self):
        reflection_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
                    "Always provide detailed recommendations, including requests for length, virality, style, etc.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))
        self.chain = reflection_prompt | llm
        self.name = "reflect"

    def __call__(self, state: MessageGraph):
        res = self.chain.invoke(
            {
                "messages": state["messages"]
            }
        )
        return {"messages": [HumanMessage(content=res.content)]}
