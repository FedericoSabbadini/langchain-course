import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from deepagents import create_deep_agent
from langchain.tools import tool
from dotenv import load_dotenv
#from tavily import TavilyClient
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama


# @tool("search")
# def search(query: str) -> str:
#     """
#     Tool function to perform a search based on the user's query. 
#     This is a placeholuvder implementation and should be replaced with actual search logic, such as querying a search engine API or a database.
    
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
    ollama_model = os.getenv("OLLAMA_MODEL", "gemma3:270m")


    agent = create_deep_agent(model = f"ollama:{ollama_model}", tools=[TavilySearch(api_key=tavily_key)]) # type: ignore

    question = "What is the weather like in Rome today?"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant that provides information about the weather using the associated tools."),
        HumanMessage(content=question)
    ])
    chain = prompt | agent
    response = chain.invoke(input={question: question})
    print(response)
