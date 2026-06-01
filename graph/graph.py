from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.nodes import generatorNodeClass, graderNodeClass, retrieverNodeClass, searcherNodeClass, routerNodeClass
from graph.states import GraphState
load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return SEARCHER
    else:
        print("---DECISION: GENERATE---")
        return GENERATOR

def grade_generation(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    if state["generation"] == "not supported":
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return GENERATOR
    elif state["generation"] == "not useful":
        print("---DECISION: GENERATION IS NOT USEFUL, RE-TRY WITH WEB SEARCH---")
        return SEARCHER
    else:
        print("---DECISION: GENERATION IS USEFUL, END---")
        return END


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")

    if state["datasource"] == "websearch":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return SEARCHER
    elif state["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVER



# --------------------------- BUILD GRAPH---------------------------
workflow = StateGraph(GraphState)
# ---------------------------- ADD NODES---------------------------
generatorNode = generatorNodeClass()
GENERATOR = generatorNode.name
graderNode = graderNodeClass()
GRADER = graderNode.name
retrieverNode = retrieverNodeClass()
RETRIEVER = retrieverNode.name
searcherNode = searcherNodeClass()
SEARCHER = searcherNode.name
routerNode = routerNodeClass()
ROUTER = routerNode.name

workflow.add_node(GENERATOR, generatorNode)
workflow.add_node(GRADER, graderNode)
workflow.add_node(RETRIEVER, retrieverNode)
workflow.add_node(SEARCHER, searcherNode)
workflow.add_node(ROUTER, routerNode)

workflow.set_entry_point(ROUTER)
# ---------------------------- EDGES ---------------------------
workflow.add_conditional_edges(
    ROUTER,
    route_question,
    {
        SEARCHER: SEARCHER,
        RETRIEVER: RETRIEVER,
    },
)

workflow.add_edge(RETRIEVER, GRADER)
workflow.add_conditional_edges(
    GRADER,
    decide_to_generate,
    {
        SEARCHER: SEARCHER,
        GENERATOR: GENERATOR,
    }
)
workflow.add_conditional_edges(
    GENERATOR,
    grade_generation,
    {
        GENERATOR: GENERATOR,
        SEARCHER: SEARCHER,
        END: END,
    }
)
workflow.add_edge(SEARCHER, GENERATOR)
workflow.add_edge(GENERATOR, END)
# ---------------------------- COMPILE & VISUALIZE---------------------------
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")