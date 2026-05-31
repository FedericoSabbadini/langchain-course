from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime 

ACTOR_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are expert researcher. Current time: {time} 
                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer.
                """
            ),
            MessagesPlaceholder(variable_name="messages"), # This allows us to pass the current messages in the state to the prompt, so that the model can use the previous messages as context for generating the answer.
            (
                "system", 
                "Answer the user's question above using the required format."
            ),
        ]
    ).partial(time=lambda: datetime.datetime.now().isoformat()) # This allows us to dynamically insert the current time into the prompt, which can be useful for time-sensitive questions or to provide context to the model about when the question is being asked.

# This allows us to reuse the same base prompt template for both the first responder and the revise node, while still being able to customize the instructions for each node.