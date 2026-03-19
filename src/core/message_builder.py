from pydantic_ai.messages import (ModelMessage, 
                                  ModelRequest, 
                                  ToolReturnPart,
                                  UserPromptPart)
import json
from data_models import ToolResult, ActionResult, VerificationResult

#------------------------------------------
#             Message Builder
#------------------------------------------
class MessageFactory:
    """Builds properly formatted messages for the Agent"""
    @staticmethod
    def build_tool_return_message(tool_result: ToolResult) -> list[ModelMessage]:
        """Construct a synthetic ToolCallPart and ToolReturnPart message pair in pydanticAI native message format"""
        tool_call_id: str = f"{tool_result.tool_name}_{id(tool_result)}"  # stable fake ID

        # The result returned to the LLM
        result_payload: str = json.dumps(
            tool_result.result_data if tool_result.success else {"error": tool_result.error_message},
            default=str,            # handles datetime, Decimal, etc.
        )
        tool_return_msg = ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=tool_result.tool_name,
                    content=result_payload,         # must be a string
                    tool_call_id=tool_call_id,      # must match the call above
                )
            ]
        )

        return [tool_return_msg]
    
    @staticmethod
    def build_error_messages(action_result: ActionResult) -> list[ModelMessage]:
        """
        Injects action error-report so the agent
        understands what it tried and why it failed.
        """
        action_error_message = ModelRequest(
                parts=[UserPromptPart(
                    content=f"[ACTION ERROR] {action_result.error_message or 'Unknown error'}. "
                            f"Please adjust your approach and retry."
                )]
            )
        
        return [action_error_message]
    
    @staticmethod
    def build_human_input_message(action_result: ActionResult) -> list[ModelMessage]:
        """Injects human input so the agent knows what the human said"""
        human_input_message = ModelRequest(
                parts=[UserPromptPart(
                    content=f"[HUMAN INPUT] {action_result.message or 'Did not receive input'}. "
                            f"Please retry."
                )]
            )
        return [human_input_message]
    
    @staticmethod
    def build_notes_message(verification_result: VerificationResult) -> list[ModelMessage]:
        message: list[ModelMessage] = [ModelRequest(
                parts=[UserPromptPart(
                    content=f"[VERIFICATION NOTES] {verification_result.notes}."
                )]
            )]
        return message