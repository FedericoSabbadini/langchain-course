import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
#from tavily import TavilyClient
from langchain_tavily import TavilySearch


# @tool("search")
# def search(query: str) -> str:
#     """
#     Tool function to perform a search based on the user's query. 
#     This is a placeholder implementation and should be replaced with actual search logic, such as querying a search engine API or a database.
    
#     Args:        
#         query (str): The search query provided by the user.
#     Returns:        
#         str: The search results or relevant information based on the query.
#     """
#     # Placeholder for search functionality
#     print(f"Searching for: {query}")
#     return tavily.search(query=query)


if __name__ == "__main__":
    load_dotenv()
    tavily_key = os.getenv("TAVILY_API_KEY")
    #tavily = TavilyClient(api_key=tavily_key) # type: ignore
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-2")

    llm = ChatAnthropic(
        model=anthropic_model, # type: ignore
        api_key=anthropic_key ) # type: ignore
    agent = create_agent(llm, tools=[TavilySearch(api_key=tavily_key)]) # type: ignore

    question = "What is the weather like in Rome today?"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that provides information about the weather using the associated tools."),
        HumanMessage(content=question)
    ])
    chain = prompt | agent
    response = chain.invoke(input={question: question})
    print(response)
