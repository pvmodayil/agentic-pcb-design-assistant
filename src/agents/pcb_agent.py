from typing import Optional, Literal, Any
from collections.abc import Sequence, Callable
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, ModelSettings, ToolOutput, ModelMessage, DeferredToolResults

from llm_model import get_llm_model
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
    action_type: Literal["proceed", "ask_human", "use_tool", "test", "update state", "conclude"] = Field(..., description="Action type to be take by the agent")
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
    human_response: Optional[str] = Field(default=None, description="Latest human feedback")
    conclusion: Optional[str] = Field(default=None, description="Final conclusion when task completed")
    
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

###########################################
# PCB Agent (Abstract agent class)
###########################################
class PCBAgentConfig(BaseModel):
    role: Literal["signal-integrity(SI)", "power-integrity(PI)", "LayerStackup"] = Field(...,description="Name of the agent")
    purpose: str = Field(...,description="Define purpose of the agent")
    mcp_servers: Sequence[str] = Field(default_factory=list, description="list of the MCP server URLs")
    tool_list: Sequence[Callable] = Field(default_factory=list, description="List of callable functions")
    tester_tools: Sequence[Callable] = Field(default_factory=list, description="List of test functions")
    previous_experience: Optional[str] = Field(None, description="Previous experience to be added in the context")
    
class PCBAgent:
    """Abstract PCB Agent Class"""
    def __init__(self, agent_config: PCBAgentConfig ) -> None:
        # Agent settings
        self.agent_config: PCBAgentConfig = agent_config
        llm_settings: LLMSettings = load_settings()
        
        # Agent context
        self.system_prompt: str = self._build_system_prompt()
        self.message_history: Sequence[ModelMessage] | None = None
        self.previous_experience: Optional[str] = self.agent_config.previous_experience
        
        # Output types
        core_agent_outputs: list[Any] = [
            AgentAction,
            AgentState, 
            TesterCall,
            FinalOutput
            ]
        
        # Agent initialise
        self.agent = Agent(
            get_llm_model(llm_settings),
            model_settings=ModelSettings(temperature=llm_settings.temperature),
            output_type=core_agent_outputs,
            system_prompt=self.system_prompt
        )
        
    def _build_system_prompt(self) -> str:
        """Basic wrapper for the agent specific system prompt"""
        
        system_prompt: str = f"""
        You are an expert '{self.agent_config.role}' agent engaged in the PCB design workflow.
        
        You must:
        - Maintain and update AgentState correctly at every step.
        - Use AgentAction to indicate whether you proceed, ask_human, use_tool, or conclude.
        - When verification of a checkpoint is needed, emit a TesterCall.
        - When the overall task is done, emit FinalOutput with success, final state, and recommendations.
        
        Now adhere to the below given purpose and complete the task:
        '{self.agent_config.purpose}'
        """
        return system_prompt
    
    def run_analysis(self, user_prompt: str, tool_result: Optional[DeferredToolResults] = None) -> AgentRunResult:
        """Synchronously run one analysis"""
        
        result: AgentRunResult = self.agent.run_sync(user_prompt=user_prompt,
                                     message_history=self.message_history,
                                     deferred_tool_results=tool_result)

        self.message_history = result.all_messages()
        
        return result
    
    def run_workflow(self, user_prompt: str) -> FinalOutput:
        current_state: AgentState = AgentState(current_checkpoint="Process Start")
        process_running: bool = True
        task: str = f"""
        Task defined by the user: {user_prompt}
        """
        while process_running:
            task += f"\n\nCurrent State: \n{current_state.model_dump_json()}"
            result: AgentRunResult = self.run_analysis(user_prompt=task)
            last_output = result.output
            
            if isinstance(last_output, FinalOutput):
                return last_output
            elif isinstance(last_output, AgentAction):
                # Handle the action types "proceed", "ask_human", "use_tool", "test", "update state", "conclude"
                pass
    
    def handle_agent_action(self, agent_action: AgentAction) -> None:
        pass