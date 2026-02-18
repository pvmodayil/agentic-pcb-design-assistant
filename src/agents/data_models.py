from typing import Optional, Literal, Any
from datetime import datetime

from enum import IntEnum

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

###########################################
# Core Data models for Agent and Workflow
###########################################
class Checkpoint(BaseModel):
    """A verified checkpoint was reached in the workflow"""
    name: str = Field(..., description="Name of the checkpoint (Unique identifier)")
    status: Literal["pending", "in_progress","failed","completed"] = Field(default="pending", description="status of this checkpoint")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional information")
    error_message: Optional[str] = Field(default=None, description="Error message in case of failure")
    
    def mark_completed(self, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mark checkpoint as completed"""
        self.status = "completed"
        self.timestamp = datetime.now()
        if metadata:
            self.metadata.update(metadata)
    
    def mark_failed(self, error: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mark checkpoint as failed"""
        self.status = "failed"
        self.error_message = error
        self.timestamp = datetime.now()
        if metadata:
            self.metadata.update(metadata)

class WorkflowState(IntEnum):
    """Workflow execution states"""
    # Success states
    COMPLETED = 200
    PARTIAL_SUCCESS = 201
    
    # Progression states
    INITIAL = 300
    ANALYZING = 301
    EXECUTING_TOOL = 302
    AWAITING_TOOL_RESULT = 303
    TOOL_COMPLETED = 304
    AWAITING_HUMAN = 305
    HUMAN_RESPONDED = 306
    
    # Testing states
    TESTING = 400
    TEST_PASSED = 401
    TEST_FAILED = 402
    
    # Error states
    ERROR = 500
    VALIDATION_ERROR = 501
    TIMEOUT = 502
    TOOL_ERROR = 503
    AGENT_ERROR = 504
    CHECKPOINT_ERROR = 505

class AgentAction(BaseModel):
    """Structured action that the agent can take"""
    action_type: Literal[
        "analyze", 
        "execute_tool", 
        "verify_checkpoint", 
        "request_human_input",
        "update_context",
        "proceed_to_next",
        "retry_checkpoint",
        "complete_workflow"
    ] = Field(..., description="Type of action to perform")
    
    # Action-specific data
    checkpoint_name: Optional[str] = Field(default=None)
    tool_name: Optional[str] = Field(default=None)
    tool_parameters: dict[str, Any] = Field(default_factory=dict)
    
    question_for_human: Optional[str] = Field(default=None)
    context_updates: dict[str, Any] = Field(default_factory=dict)
    
    reasoning: str = Field(..., description="Why this action is being taken")
    expected_outcome: Optional[str] = Field(default=None)

class ToolDefinition(ABC,BaseModel):
    """Definition of a tool that can be used by the agent
       Each tool must implement a validate_parameters function     
    """
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What the tool does")
    parameters_schema: dict[str, Any] = Field(default_factory=dict, description="JSON schema for tool parameters")
    is_async: bool = Field(default=False, description="Whether tool execution is async")
    requires_human_approval: bool = Field(default=False)
    can_fail: bool = Field(default=True, description="Whether tool failure is recoverable")
    
    @abstractmethod
    def validate_parameters(self, parameters: dict[str,Any]) -> Optional[list[str]]:
        """Validate the paramters for this tool (schema and rules) return None if valid"""
        pass

class ToolResult(BaseModel):
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)
   
class WorkflowResult(BaseModel):
    """Final result of workflow execution"""
    success: bool
    session_id: str
    workflow_type: str
    final_state: WorkflowState
    
    completed_checkpoints: list[Checkpoint]
    failed_checkpoints: list[Checkpoint]
    
    results: dict[str, Any] = Field(default_factory=dict)
    recommendations: str|None = Field(default=None, description="Recommendations from the agent")
    summary: str = Field(..., description="Executive summary")
    
    total_execution_time: float = Field(default=0.0)
    errors: list[str] = Field(default_factory=list)

class ActionResult(BaseModel):
    status: Literal[
        "analyzed",
        "context_updated",
        "tool_executed",
        "checkpoint_verified",
        "human_input_requested",
        "workflow_completed",
        "retry_required",
        "verification_failed",
        "error"] = Field(..., description="status")
    tool_result: Optional[ToolResult] = Field(default=None, description="Results from tool executions")
    checkpoint: Optional[str] = Field(default=None, description="Name of the checkpoint, if verification")
    error_message: Optional[str] = Field(default=None, description="Error message")
    message: Optional[str] = Field(default=None, description="Message")

class AgentState(BaseModel):
    """Tracks the current state of agent execution"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    version: int = Field(default=1, description="State schema version")
    workflow_state: WorkflowState = Field(default=WorkflowState.INITIAL)
    current_checkpoint: Optional[str] = Field(default=None, description="Current checkpoint name")
    completed_checkpoints: list[str] = Field(default_factory=list)
    pending_checkpoints: list[str] = Field(default_factory=list)
    
    # Context data
    context_data: dict[str, Any] = Field(default_factory=dict, description="Workflow-specific context")
    tool_results: Optional[ToolResult] = Field(default=None, description="Results from tool executions")
    
    # Human interaction
    needs_human_input: bool = Field(default=False)
    human_question: Optional[str] = Field(default=None)
    human_response: Optional[str] = Field(default=None)
    
    # Error tracking
    errors: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    
    def can_retry(self) -> bool:
        """Check if retry is allowed"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment retry counter"""
        self.retry_count += 1
        logger.warning(f"Retry count incremented to {self.retry_count}/{self.max_retries}")

class Summary(BaseModel):
    summary: str = Field(...,description="Entire workflow summary")
    recommendation: str = Field(..., description="Recommendations for the designer")