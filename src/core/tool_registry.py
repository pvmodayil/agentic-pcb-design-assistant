from typing import Any, Callable
from loguru import logger

from data_models import ToolDefinition
#--------------------------------------------
# Tool Registry (To Register Tools for Agent)
#--------------------------------------------

ToolFunction = Callable[..., Any]

class ToolRegistry:
    """
    Registry for managing agent tools
    """
    
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}
        self._tool_functions: dict[str, ToolFunction] = {}
    
    def register_tool(self, tool_def: ToolDefinition, tool_func: ToolFunction) -> None:
        """Register a tool with its implementation"""
        self._tools[tool_def.name] = tool_def
        self._tool_functions[tool_def.name] = tool_func
        logger.info(f"Registered tool: {tool_def.name}")
    
    def get_tool_definition(self, tool_name: str) -> ToolDefinition:
        """Get tool definition by name"""
        return self._tools.get(tool_name) #type:ignore
    
    def get_tool_function(self, tool_name: str) -> ToolFunction:
        """Get tool implementation by name"""
        return self._tool_functions.get(tool_name) #type:ignore
    
    def list_tools(self) -> list[str]:
        """List all registered tool names"""
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> str:
        import json
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(json.dumps(tool.parameters_schema, indent=2))
        return "\n\n".join(descriptions)