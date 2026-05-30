# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers
    
    Args:
        a: The first number
        b: The second number
        
    Returns:
        The sum of the two numbers
    """
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers
    
    Args:
        a: The first number
        b: The second number
        
    Returns:
        The product of the two numbers
    """
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
    # stdio is a transport protocol that allows the server to communicate with 
    # the client through standard input and output. This is useful for our 
    # math server because it allows us to easily run the server as a subprocess
    #  and communicate with it through the command line.

    # to run the server, we can use the command: python math_server.py