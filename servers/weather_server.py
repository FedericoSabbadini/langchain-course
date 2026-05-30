from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location.
    
    Args:
        location: The location for which to get weather
        
    Returns:
        A description of the weather at the specified location
    """
    return "Hot as hell"

if __name__ == "__main__":
    mcp.run(transport="sse")
    # sse is a transport protocol that allows the server to push updates to 
    # the client in real time. This is useful for our weather server because 
    # we can send updates to the client whenever the weather changes without
    #  the client having to request it.

    # to run the server, we can use the command: python weather_server.py