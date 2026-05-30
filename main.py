import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client 
# stdio_client is a client that allows us to communicate with a server 
# that uses the stdio transport protocol.

load_dotenv()
import sys
model = os.getenv("ANTHROPIC_MODEL", "claude-2")
key = os.getenv("ANTHROPIC_API_KEY")
ollama_model = os.getenv("OLLAMA_MODEL")
llm = ChatAnthropic(model=model, anthropic_api_key=key)

# info about the stdio server we want to connect to. In this case, it's our math server. 
# We specify the command to run the server and the arguments to pass to the command. 
# This will allow us to run the math server as a subprocess and communicate with it through the command line.
stdio_server_params = StdioServerParameters(
    command=sys.executable, # this is the command to run the server. We use sys.executable to get the path to the Python interpreter, which allows us to run the server as a Python script.
    #     command="python", # language to run the server
    args=["C:\\Users\\Federico\\DesktopW\\langchain-course\\servers\\math_server.py"], # path to the server file
)

async def main():
    async with stdio_client(stdio_server_params) as (read,write):    
        # stdio_client is an async context manager that runs the server as a subprocess and provides us with read and write streams to communicate with the server.
        async with ClientSession(read_stream=read, write_stream=write) as session:
            # CLientSession is an async context manager that provides us with a session to communicate with the server. We pass the read and write streams from the stdio_client to the ClientSession.
            await session.initialize()
            print("session initialized")
            tools = await load_mcp_tools(session)


            agent = create_agent(model=f"ollama:{ollama_model}", tools=tools)

            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 54 + 2 * 3?")]})
            print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())

