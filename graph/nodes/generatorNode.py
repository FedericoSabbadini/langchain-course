from typing import Any, Dict
from graph.states import GraphState
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
from langsmith import Client

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score for relevance check on generated answer to question."""
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")


class generatorNodeClass:
    def __init__(self) -> None:
        llm = ChatOllama(model = os.getenv("OLLAMA_MODEL"), temperature=0)
        client = Client()
        prompt = client.pull_prompt("rlm/rag-prompt",     dangerously_pull_public_prompt=True
) # this is a prompt that we created and uploaded to the langsmith hub, you can find it in the folder graph/prompts/ragPrompt.py

        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question \n 
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_answer),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
)

        structured_llm_grader = llm.with_structured_output(GradeHallucinations)
        structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)
        
        self.hallucination_chain = hallucination_prompt | structured_llm_grader
        self.answer_chain = answer_prompt | structured_llm_answer_grader
        self.generation_chain = prompt | llm | StrOutputParser()
        self.name = "generate"


    def __call__(self, state: GraphState) -> Dict[str, Any]:
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.generation_chain.invoke({"context": documents, "question": question})
        
        score = self.hallucination_chain.invoke(
            {"documents": documents, "generation": generation}
        )

        if hallucination_grade := score.binary_score:
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            print("---GRADE GENERATION vs QUESTION---")
            answer_grade = self.answer_chain.invoke({"question": question, "generation": generation})
            if answer_score := answer_grade.binary_score:
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                generation = "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            generation = "not supported"


        return {"documents": documents, "question": question, "generation": generation}
    


