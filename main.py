from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.messages import HumanMessage
from nodes.toolNode import toolNode
from nodes.draftNode import draftNodeClass
from nodes.reviseNode import reviseNodeClass

MAX_ITERATIONS = 2



def should_continue(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"




# --------------------------------- Graph Construction ---------------------------------
builder = StateGraph(MessagesState)
# -------------------------------- Add nodes ----------------------------------
draftNode = draftNodeClass()
reviseNode = reviseNodeClass()
builder.add_node(draftNode.name, draftNode.__call__)
builder.add_node("execute_tools", toolNode)
builder.add_node(reviseNode.name, reviseNode.__call__)
builder.add_edge(START, draftNode.name)
builder.add_edge(draftNode.name, "execute_tools")
builder.add_edge("execute_tools", reviseNode.name)
builder.add_conditional_edges(reviseNode.name, should_continue, 
                              [
                                  "execute_tools", 
                                  END
                            ])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())



res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)
# Extract the final answer from the last message with tool calls
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])
print(res)



if __name__ == "__main__":
    print("Hello LangGraph")

    question = "Write about AI-Powered SOC / autonomous soc  problem domain,  list startups that do that and raised capital."
    inputs = {
        "messages": [
            HumanMessage(content=question)
        ]
    }
    response = graph.invoke(inputs)
    print(response)