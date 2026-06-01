from pprint import pprint
from dotenv import load_dotenv
load_dotenv()
from graph.nodes.retrieverNode import retrieverNodeClass
from graph.nodes.graderNode import GradeDocuments, graderNodeClass
from graph.nodes.generatorNode import generatorNodeClass
from graph.nodes.routerNode import RouteQuery, routerNodeClass
from graph.nodes.graderNode import GradeHallucinations

retriever = retrieverNodeClass()
grader = graderNodeClass()
generator = generatorNodeClass()
question_router = routerNodeClass()

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.__call__({"question": question})["documents"]
    doc_txt = docs[1].page_content

    res: GradeDocuments = grader.invoke(
        {"question": question, "document": doc_txt}
    ) # this sintax res: means that we are expecting the output of the grader.invoke 
    # to be of type GradeDocuments, which is a pydantic model we defined in graderNode.py

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.__call__({"question": question})["documents"]
    doc_txt = docs[1].page_content

    res: GradeDocuments = grader.invoke(
        {"question": "how to make pizza", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.__call__({"question": question})["documents"]
    generation = generator.__call__({"question": question, "documents": docs})["generation"]
    print(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.__call__({"question": question})["documents"]

    res: GradeHallucinations = generator.__call__(
        {"documents": docs, "question": question}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.__call__({"question": question})["documents"]

    res: GradeHallucinations = generator.__call__(
        {
            "documents": docs,
            "question": question,
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.__call__({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.__call__({"question": question})
    assert res.datasource == "websearch"