from typing import Optional, Literal
from llm_model import get_llm_model
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, ToolOutput

from config.settings import load_settings, LLMSettings

###########################################
# Data models for the agent
###########################################
class Checkpoint(BaseModel):
    """A verified checkpoint was reached"""
    name: str = Field(..., description="Name of the checkpoint")
    status: Literal["pending", "completed"] = Field(..., description="status of this checkpoint")
    timestamp: datetime = Field(default_factory=datetime.now)
    
class AgentAction(BaseModel):
    """List of possible actions for the agent"""
    action_type: Literal["proceed", "ask_human", "use_tool", "conclude"] = Field(..., description="Action type to be take by the agent")
    checkpoint_name: Optional[str] = Field(None, description="Target checkpoint for proceed/use_tool")
    question_to_human: Optional[str] = Field(None, description="Question for human feedback")
    tool_name: Optional[str] = Field(None, description="Tool to call")
    tool_args: Optional[dict] = Field(None, description="Arguments for the tool")

class AgentState(BaseModel):
    """Current state tracking for the agent."""
    current_checkpoint: str = Field(..., description="Name of current checkpoint")
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    actions_history: list[AgentAction] = Field(default_factory=list)
    needs_human_input: bool = Field(default=False, description="Flag for human feedback required")
    human_response: Optional[str] = Field(None, description="Latest human feedback")
    conclusion: Optional[str] = Field(None, description="Final conclusion when complete")
    # Agent state should have messages/chathistory too
    
class TesterCall(BaseModel):
    """Structured call to verification tester."""
    target_checkpoint_name: str = Field(..., description="checkpoint to verify")
    verification_data: Optional[dict] = Field(default={}, description="Data for tester")

class FinalOutput(BaseModel):
    """Final structured output when agent completes."""
    success: bool = Field(..., description="Whether all checkpoints completed successfully")
    state: AgentState = Field(..., description="Final agent state")
    summary: str = Field(..., description="Executive summary of results")
    recommendations: Optional[list[str]] = Field(default_factory=list)

llm_settings: LLMSettings = load_settings()
agent = Agent(
    get_llm_model(llm_settings),
    model_settings=ModelSettings(temperature=llm_settings.temperature),
    output_type=[
        ToolOutput(AgentAction, name="next_action", description="Next agent action"),
        ToolOutput(AgentState, name="state_update", description="State update"),
        ToolOutput(TesterCall, name="tester_call", description="Verification request"),
        ToolOutput(FinalOutput, name="final_result", description="Final completion")
    ]
)