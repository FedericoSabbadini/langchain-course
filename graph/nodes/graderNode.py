from typing import Any, Dict
from graph.states import GraphState
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")




class graderNodeClass:
    def __init__(self):
        llm = ChatOpenAI(temperature=0)
        structured_llm_grader = llm.with_structured_output(GradeDocuments)
        
        system_grader = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_grader),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self.chain = grade_prompt | structured_llm_grader
        self.name = "grade_documents"


    def __call__(self, state: GraphState) -> Dict[str, Any]:
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = False
        for d in documents:
            score = self.chain.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = True
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}