from typing import List

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "Hot as hell"

if __name__ == "__main__":
    mcp.run(transport="sse")
    # sse is a transport protocol that allows the server to push updates to 
    # the client in real time. This is useful for our weather server because 
    # we can send updates to the client whenever the weather changes without
    #  the client having to request it.