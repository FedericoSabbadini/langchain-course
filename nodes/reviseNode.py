from .toolNode import ReviseAnswer
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from langgraph.graph import MessagesState
from langgraph.graph import MessagesState
from prompt import ACTOR_PROMPT_TEMPLATE

load_dotenv()


class reviseNodeClass:
    def __init__(self):
        revise_instructions = """
            Revise your previous answer using the new information.
                - You should use the previous critique to add important information to your answer.
                    - You MUST include numerical citations in your revised answer to ensure it can be verified.
                    - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
                        - [1] https://example.com
                        - [2] https://example.com
                - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
        """

    

        revisor_prompt_template = ACTOR_PROMPT_TEMPLATE.partial( first_instruction=revise_instructions)

        llm = ChatOllama(model = os.getenv("OLLAMA_MODEL", "langgraph-demo"))
        llm_with_tools = llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

        self.chain = (
            revisor_prompt_template 
            | llm_with_tools
        )
        self.name = "revise"

    def __call__(self, state: MessagesState):
        res = self.chain.invoke(
            {
                "messages": state["messages"]
            }
        )
        return {"messages": [res]}