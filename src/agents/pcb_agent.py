from typing import Optional, Literal, Any
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResult, ModelSettings, ModelMessage, DeferredToolResults

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

class Tool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]  

class AgentAction(BaseModel):
    """List of possible actions for the agent"""
    action_type: Literal["proceed", "ask_human", "use_tool", "test", "update state", "conclude"] = Field(..., description="Action type to be take by the agent")
    checkpoint_name: Optional[str] = Field(None, description="Target checkpoint for proceed/use_tool")
    question_to_human: Optional[str] = Field(None, description="Question for human feedback")
    tool_call: Optional[Tool]
    new_state: Optional[AgentState]

class AgentState(BaseModel):
    """Current state tracking for the agent."""
    version: int =Field(default=1, description="Version of the state")
    current_checkpoint: str = Field(..., description="Name of current checkpoint")
    checkpoints: list[Checkpoint] = Field(default_factory=list)
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
# Memory Agent
###########################################
class MEMAgent:
    """Memory management agent for relevant context generation"""
    def __init__(self, workflow: str) -> None:
        system_prompt: str = f"""
        You are an expert PCB design assistant engaged in the '{workflow}' workflow. You main task is to assist the process by contextualizing 
        relevant information from the entire message log. You must focus only on the most relevant infromation that will help with the query.
        """
        self.mem_agent = Agent(
            get_llm_model(llm_settings=load_settings(key="mem_llm")),
            model_settings=ModelSettings(temperature=0.3),
            system_prompt=system_prompt
        )
    
    def _relevant_context(self, query: str, message_history: list[ModelMessage]) -> list[ModelMessage]:
        memory_prompt: str =f"""
        Current query: {query}
        
        From recent conversation, create a COMPACT memory state (200 words max) that:
        1. Summarizes KEY insights/decisions relevant to this query
        2. Notes any ERRORS/feedback from previous steps  
        3. Lists ACTIVE context (current checkpoint, open issues)
        4. Forgets irrelevant details 
        """
        context: list[ModelMessage] = self.mem_agent.run_sync(user_prompt=memory_prompt,
                                               message_history=message_history).new_messages()
        
        return context
        
        
###########################################
# PCB Agent (Abstract agent class)
###########################################
class PCBAgentConfig(BaseModel):
    role: Literal["signal-integrity(SI)", "power-integrity(PI)", "LayerStackup"] = Field(...,description="Name of the agent")
    purpose: str = Field(...,description="Define purpose of the agent")
    mcp_servers: Sequence[str] = Field(default_factory=list, description="list of the MCP server URLs")
    tool_list: dict[str, Tool] = Field(default_factory=dict, description="List of callable functions")
    tester_tools: dict[str, Tool] = Field(default_factory=dict, description="List of test functions")
    previous_experience: Optional[str] = Field(None, description="Previous experience to be added in the context")

class PCBAgent:
    """Abstract PCB Agent Class"""
    def __init__(self, agent_config: PCBAgentConfig ) -> None:
        # Memory Agent
        self.mem_agent: MEMAgent = MEMAgent(workflow=self.agent_config.role)
        
        # Agent settings
        self.agent_config: PCBAgentConfig = agent_config
        llm_settings: LLMSettings = load_settings(key="llm")
        
        # Agent context
        self.system_prompt: str = self._build_system_prompt()
        self.message_history: list[ModelMessage] = []
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
    
    def run_analysis(self, query: str, tool_result: Optional[DeferredToolResults] = None) -> AgentRunResult:
        """Synchronously run one analysis"""
        
        result: AgentRunResult = self.agent.run_sync(user_prompt=query,
                                     message_history=self.mem_agent._relevant_context(query=query,message_history=self.message_history),
                                     deferred_tool_results=tool_result)

        self.message_history.extend(result.new_messages())
        
        return result