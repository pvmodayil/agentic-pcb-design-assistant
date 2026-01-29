import uuid
from typing import Optional, Any
from abc import ABC, abstractmethod
from enum import IntEnum
from pydantic import BaseModel, Field
from pcb_agent import Checkpoint, Tool

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
    
class WorkFlowContext(BaseModel):
    session_id: str
    current_checkpoint: Checkpoint
    history: list[Checkpoint] = Field(default_factory=list)
    max_retries: int = 3
    retry_count: int = 0
        
class WorkflowConfig(BaseModel):
    "Configuration for workflow"
    workflow_type: str = Field(...,description="SI, PI, or LayerStackup")
    purpose: str = Field(...,description="Workflow objective")
    checkpoints: list[Checkpoint] = Field(default_factory=list,description="Ordered list of Checkpoints")
    tools: list[Tool] = Field(default_factory=list,description="List of Tool")

class Workflow(ABC):
    def __init__(self, config: WorkflowConfig) -> None:
        self.config: WorkflowConfig = config
        self.current_state = WorkflowState.INITIAL
        
        initial_checkpoint = Checkpoint(name="INITIAL", status='completed')
        self.workflow_context = WorkFlowContext(session_id=str(uuid.uuid4()),
                                                current_checkpoint=initial_checkpoint)
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
    
    def get_current_state(self) -> None:
        """
        Serialisable state information
        Returns
        -------
        dict[str,Any]
        """
        pass
        
    
        