from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from .toolNode import AnswerQuestion
load_dotenv()
from langgraph.graph import MessagesState
from prompt import ACTOR_PROMPT_TEMPLATE


class draftNodeClass:
    def __init__(self):

        first_responder_prompt_template = ACTOR_PROMPT_TEMPLATE.partial( first_instruction="Provide a detailed ~250 word answer.")

        llm = ChatOllama(model = os.getenv("OLLAMA_MODEL", "langgraph-demo"))
        llm_with_tools = llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")


        self.chain = (
            first_responder_prompt_template 
            | llm_with_tools
        )
        self.name = "draft"

    def __call__(self, state: MessagesState):
        """
            Invoke the generation chain with the current state.
            It will be used in the Node of the grafh without (), so it needs to be a callable that takes the state as input.
        """
        res = self.chain.invoke(
            {
                "messages": state["messages"]
            }
        )
        return {"messages": [res]}
