from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
load_dotenv()


@tool
def triple(num: float) -> float:
    """
    Triples a number.

    Args:
        num (float): The number to triple.
    Returns:
        float: The tripled number.
    """
    return 3 * float(num)

tools = [TavilySearch(max_results=10), triple]

tool_node = ToolNode(tools)