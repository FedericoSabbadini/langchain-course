from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from state import MessageGraph
from nodes.generationNode import generationNodeClass
from nodes.reflectionNode import reflectionNodeClass


def should_continue(state: MessageGraph):
    """
        Determines whether to continue the loop based on the number of messages.
        If there are more than 6 messages, it ends the loop; otherwise, it continues to the reflection node.
        Else it continues to the reflection node.
    """
    if len(state["messages"]) > 6:
        return END
    return reflectChain_node.name


# ------------------------------- Graph Construction -------------------------------
builder = StateGraph(state_schema=MessageGraph)
# -------------------------------- Nodes -------------------------------
reflectChain_node = reflectionNodeClass()
generateChain_node = generationNodeClass()
builder.add_node(generateChain_node.name, generateChain_node.__call__)
builder.add_node(reflectChain_node.name, reflectChain_node.__call__)
builder.set_entry_point(reflectChain_node.name)
# -------------------------------- Edges -------------------------------
builder.add_conditional_edges(generateChain_node.name, should_continue)
builder.add_edge(reflectChain_node.name, generateChain_node.name)

# -------------------------------- Compile and Visualize -------------------------------
graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()


if __name__ == "__main__":
    print("Hello LangGraph")

    question = "Make this tweet better: @LangChainAI — newly Tool Calling feature is seriously underrated. After a long wait, it's here- making the implementation of agents across different models with function calling - super easy. Made a video covering their newest blog post"
    inputs = {
        "messages": [
            HumanMessage(content=question)
        ]
    }
    response = graph.invoke(inputs)
    print(response)