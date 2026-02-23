# mcp_tool_adapter.py
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.types import CallToolResult
from data_models import ToolResult
from datetime import datetime

class MCPToolAdapter:
    """
    Wraps an external MCP server tool as a plain async callable.
    Registered into ToolRegistry exactly like any internal tool.
    """
    def __init__(self, server_config: StdioServerParameters | str) -> None:
        """
        Initialise the wrapper

        Parameters
        ----------
        server_config : StdioServerParameters | str
            Takes either a StdioServerParameters (for local subprocess tools) or a plain string URL (for remote HTTP tools).
        """
        self.server_config: StdioServerParameters | str = server_config
    
    async def call(self, tool_name: str, parameters: dict) -> ToolResult:
        start = datetime.now()
        try:
            # Determine which transport to use
            if isinstance(self.server_config, StdioServerParameters):
                ctx = stdio_client(self.server_config)
            else:
                ctx = sse_client(self.server_config)
            
            async with ctx as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    mcp_result: CallToolResult = await session.call_tool(tool_name, parameters)
                    
                    result_data = {}
                    for block in mcp_result.content:
                        if text := getattr(block, "text", None):
                            result_data["text"] = text
                        if data := getattr(block, "data", None):
                            result_data["data"] = data
                    
                    return ToolResult(
                        tool_name=tool_name,
                        success=not mcp_result.isError,
                        result_data=result_data,
                        execution_time=(datetime.now() - start).total_seconds(),
                        error_message=result_data.get("text") if mcp_result.isError else None
                    )
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result_data={},
                execution_time=(datetime.now() - start).total_seconds(),
                error_message=str(e)
            )