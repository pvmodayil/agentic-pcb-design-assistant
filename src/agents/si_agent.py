from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

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

class AgentOutput(BaseModel):
    """Union output type for Agent class - handles all possible responses."""
    __discriminator_value__: str = "type"
    type: Literal["action", "state_update", "tester_call", "final"] = Field(..., discriminator='type')
    
    # Discriminated union fields
    action: Optional[AgentAction] = Field(None, description="Next action to take")
    state: Optional[AgentState] = Field(None, description="Updated state")
    tester: Optional[TesterCall] = Field(None, description="Tester verification request")
    final: Optional[FinalOutput] = Field(None, description="Final completion")

ollama_model = OpenAIChatModel(
    model_name='gemma3',
    provider=OllamaProvider(base_url='http://localhost:11434/v1')
)

agent = Agent(ollama_model, output_type=AgentOutput)