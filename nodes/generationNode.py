from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from state import MessageGraph
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

load_dotenv()


class generationNodeClass:
    def __init__(self):
        generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
                    " Generate the best twitter post possible for the user's request."
                    " If the user provides critique, respond with a revised version of your previous attempts.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        llm = ChatOllama(model = os.getenv("OLLAMA_MODEL", "langgraph-demo"))

        self.chain = generation_prompt | llm
        self.name = "generate"

    def __call__(self, state: MessageGraph):
        """
            Invoke the generation chain with the current state.
            It will be used in the Node of the grafh without (), so it needs to be a callable that takes the state as input.
        """
        res = self.chain.invoke(
            {
                "messages": state["messages"]
            }
        )
        return {"messages": [HumanMessage(content=res.content)]}
