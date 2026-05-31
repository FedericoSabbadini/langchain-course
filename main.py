from dotenv import load_dotenv
load_dotenv()
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from nodes.llmNode import agentReason_node
from nodes.toolNode import tool_node 


AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# -------------------- Graph Construction --------------------
flow = StateGraph(MessagesState)
# ----------Nodes ----------
flow.add_node(AGENT_REASON, agentReason_node)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)
# ----------Edges ----------
flow.add_conditional_edges(
    AGENT_REASON,
    should_continue,
    {
        END: END,
        ACT: ACT,
    }, # optional parameter to specify the default edge if the condition function returns a value not in the mapping
       # but they are useful to catch errors in the condition function should_continue
)
flow.add_edge(ACT, AGENT_REASON)

# -------------------- Graph Compilation and Execution ----------------
app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")



if __name__ == "__main__":
    print("Hello ReAct with LangGraph")
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="what is the average temperature in winter in Tokyo? List it and then Triple it "
                )
            ]
        }
    )
    print(res["messages"][LAST].content)