from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
import os
from langsmith import traceable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# HumanMessage is a message from the user
# SystemMessage is a message from the system
# ToolMessage is a message that represents a tool used and its output.


@tool("get_product_price", return_direct=True)
def get_product_price(product_name: str) -> str:
    """Look up the price of a product in the catalog."""
    print(f"    Getting price for {product_name}...")
    prices = {
        "laptop": "$999",
        "smartphone": "$499",
        "headphones": "$199"
    }
    return prices.get(product_name.lower(), "Product not found")

@tool("apply_discount", return_direct=True)
def apply_discount(price: str, discount_tier: str) -> str:
    """Apply a discount to a product price."""
    print(f"    Applying {discount_tier}% discount to {price}...")
    tiers = {
        "bronze": 0.10,
        "silver": 0.20,
        "gold": 0.30
    }
    discount = tiers.get(discount_tier.lower())
    if discount is None:
        return "Invalid discount tier"
    price_value = float(price.strip("$"))
    discounted_price = price_value * (1 - discount)
    return f"${discounted_price:.2f}"

@traceable(name="agent_execution")
def run_agent(question: str) -> str:
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}
    llm = init_chat_model(model="ollama: " + model)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for i in range (MAX_ITERATION): 
        print(f"Iteration {i+1}...")
        aimessage = llm_with_tools.invoke(messages)
        messages.append(aimessage)
        tool_calls = aimessage.tool_calls

        if not tool_calls:
            return aimessage.content
        else:
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args")

            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool {tool_name} not found")
            
            observation = tool_to_use.invoke(tool_args)
            messages.append(
                ToolMessage(content=observation, tool_call_id=tool_call.get("id"))
            )
            print(f"    Tool {tool_name} returned: {observation}")


if __name__ == "__main__":
    load_dotenv()
    MAX_ITERATION = 3
    model = os.getenv("OLLAMA_MODEL", "gpt-4-0613")

    question = 'What is the price of the product named "laptop"?'
    answer = run_agent(question)
    print(f"    Final answer: {answer}")