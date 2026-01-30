import uuid
from typing import Optional, Any
from abc import ABC, abstractmethod
from enum import IntEnum
from pydantic import BaseModel, Field
from pcb_agent import Checkpoint, Tool, AgentAction, PCBAgent
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger: logging.Logger = logging.getLogger(__name__)
class WorkflowState(IntEnum):
    # Success
    COMPLETED = 200
    PARTIAL_SUCCESS = 201
    
    # Workflow progression
    INITIAL = 300
    ANALYSING = 301
    TOOL_CALL = 302
    TOOL_PENDING = 303  # Tool running
    TOOL_SUCCESS = 304
    HUMAN_INPUT = 305
    
    # Testing
    TESTING = 400
    TEST_PASS = 401
    TEST_FAIL = 402
    
    # Errors
    ERROR = 500
    VALIDATION_ERROR = 501
    TIMEOUT = 502
    TOOL_ERROR = 503
    AGENT_ERROR = 504
class Result(BaseModel):  
    success: bool
    session_id: str
    final_state: WorkflowState
    checkpoints: list[Checkpoint]
    results: dict[str,Any]
class WorkFlowContext(BaseModel):
    session_id: str
    target_checkpoint: Checkpoint
    history: list[Checkpoint] = Field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
        
class WorkflowConfig(BaseModel):
    "Configuration for workflow"
    workflow_type: str = Field(...,description="SI, PI, or LayerStackup")
    purpose: str = Field(...,description="Workflow objective")
    checkpoints: list[str] = Field(default_factory=list,description="Ordered list of Checkpoints")
    current_checkpoint_idx: int = Field(default=0,description="Id of the current checkpoint")
    tools: list[Tool] = Field(default_factory=list,description="List of Tool")

class Workflow(ABC):
    def __init__(self, config: WorkflowConfig, initial_prompt, agent: PCBAgent) -> None:
        # Agent
        self.agent: PCBAgent = agent
        self.next_action: AgentAction
        
        # Human input
        self.human_intervention: str
        self.initial_prompt: str
        
        # State
        self.state: WorkflowState = WorkflowState.INITIAL
        # Initialise start points
        self.config: WorkflowConfig = config
        self.current_state = WorkflowState.INITIAL
        
        # Set up the first target checkpoint into the context
        target_checkpoint: Checkpoint = Checkpoint(name=self.config.checkpoints[0],
                                                   status='pending')
        self.workflow_context = WorkFlowContext(session_id=str(uuid.uuid4()),
                                                target_checkpoint=target_checkpoint)
        
        self.last_safe_checkpoint: Checkpoint
        self.action_results: dict[str,Any] = {}
    
    def get_next_target_checkpoint(self) -> Checkpoint:
        """
        Gets the next logical checkpoint from the ordered list
        
        Returns
        -------
        Checkpoint
            Next target checkpoint with status 'pending'
        """
        if self.config.current_checkpoint_idx + 1 < len(self.config.checkpoints):
            self.config.current_checkpoint_idx += 1
    
        return Checkpoint(name=self.config.checkpoints[self.config.current_checkpoint_idx],
                          status='pending')
    
    def reset_to_checkpoint(self, checkpoint_name: str) -> Checkpoint:
        """
        Resets to a known checkpoint

        Parameters
        ----------
        checkpoint_name : str
            name of the checkpoint

        Returns
        -------
        Checkpoint
            Next target checkpoint with status 'pending'
        """
        if checkpoint_name in self.config.checkpoints:
            self.config.current_checkpoint_idx = self.config.checkpoints.index(checkpoint_name)
            self.workflow_context.retry_count += 1
            logger.info(f"Reset to the checkpoint: {checkpoint_name}, retry count: {self.workflow_context.retry_count}")
        else:
            logger.warning(f"Unknown checkpoint: {checkpoint_name}")
            self.config.current_checkpoint_idx = self.config.checkpoints.index(self.last_safe_checkpoint.name)
            self.workflow_context.retry_count += 1
            
        return self.get_next_target_checkpoint()
    
    def get_current_state(self) -> None:
        """
        Serialisable state information
        Returns
        -------
        dict[str,Any]
        """
        pass
    
    def _transition_to_state(self):
        pass
        
    def step(self, action: AgentAction) -> dict[str, Any]:
        action_type: str = action.action_type
        try:
            if action_type == "ask_human":
                return {}
            if action_type == "use_tool":
                return {}
            if action_type == "test":
                return {}
            if action_type == "proceed":
                return {}
            if action_type == "update_state":
                return {}
            if action_type == "conclude":
                return {}
            else:
                raise ValueError(f"Unknown action {action}")
        except Exception as e:
            self.state = WorkflowState.AGENT_ERROR
            return {
                "state": self.state,
                "info": f"Step function encountered unknow action {action_type}"
            }
    
    def get_final_results(self) -> Result:
        """Get complete workflow results."""
        return Result(success=self.current_state in [WorkflowState.COMPLETED, WorkflowState.PARTIAL_SUCCESS],
                      session_id=self.workflow_context.session_id,
                      final_state=self.current_state,
                      checkpoints=self.workflow_context.history,
                      results=self.action_results)
        
    @abstractmethod
    def execute_tool(self, tool_name: str) -> None:
        """
        Fucntion to execute the required tool, parameters and outputs are to be linked using Tool dtype
        """
        pass
    
    @abstractmethod
    def verify_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Verify the checkpoint, execute tester fucntions"""
        pass  