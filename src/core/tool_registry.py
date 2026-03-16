from typing import Any, Callable
import inspect
from loguru import logger

from data_models import ToolDefinition
from data_models import ToolResult
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
        """json formatted description of tools"""
        import json
        descriptions = []
        for name, tool in self._tools.items():
            descriptions.append(json.dumps(tool.parameters_schema, indent=2))
        return "\n\n".join(descriptions)
    
    async def handle_tool_call(self, tool_name: str, tool_parameters: dict[str,Any]) -> ToolResult:
        """handle the tool call with validations"""
        error_message: str | None = None
        
        tool_func: ToolFunction|None = self.get_tool_function(tool_name)
        if not tool_func:
            error_message = f"Unknown tool: {tool_name}"
        
        tool_def: ToolDefinition = self.get_tool_definition(tool_name)
        if not tool_def:
            error_message = f"Tool definition missing: {tool_name}"
        
        # Validate parameters (custom validation logic for every tool)
        if tool_def and not error_message:
            schema_errors: list|None = tool_def.validate_parameter_schema(tool_parameters)
            if schema_errors:
                error_message = f"Tool '{tool_name}' parameters failed schema validation with errors: {schema_errors}"
        
        if tool_def and not error_message:
            parameter_errors: list|None = tool_def.validate_parameters(tool_parameters)
            if parameter_errors:
                error_message = f"Tool '{tool_name}' parameters failed validation with errors: {parameter_errors}"
        
        if not error_message:
            try:
                logger.info(f"Executing tool: {tool_name}")
                if inspect.iscoroutinefunction(tool_func):
                    result: dict[str,Any] = await tool_func(**tool_parameters)
                else:
                    result: dict[str,Any] = tool_func(**tool_parameters)
                
                # Store result
                tool_result = ToolResult(
                    tool_name=tool_name,
                    success=True,
                    result_data=result,
                )
                
                logger.success(f"Executed tool: {tool_name}")
                return tool_result
            except Exception as e:
                logger.error(f"Tool execution failed: {e}", exc_info=True)
                error_message=f"Tool {tool_name} execution failed: {e}"
        
        # Return error result
        tool_result = ToolResult(
            tool_name=tool_name,
            success=False,
            error_message=error_message
        )
        return tool_result
        
            
        
            