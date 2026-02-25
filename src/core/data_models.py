from typing import Optional, Literal, Any
from datetime import datetime

from jsonschema import validate, ValidationError

from enum import IntEnum

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger

#---------------------------------------------------------
#                     Agent & Workflow
#---------------------------------------------------------
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

#---------------------------------------------------------
#                       Tools
#---------------------------------------------------------
class ToolParameter(BaseModel):
    """Tool parameter definition (JSON Schema compatible)."""
    name: str
    type: Literal["number", "string", "boolean", "object", "array"]
    description: str
    required: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None
    
class ToolDefinition(ABC,BaseModel):
    """Definition of a tool that can be used by the agent
       Each tool must implement a validate_parameters function     
    """
    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="What the tool does")
    category: Literal[
            "io",            # file/db/FS I/O
            "network",       # HTTP, APIs
            "retrieval",     # search, RAG, knowledge lookup
            "calculation",   # math, scoring, analytics
            "simulation",    # simulation tools   
            "code",          # code execution, linting, generation
            "monitoring",    # logs, metrics, alerts
            "orchestration", # scheduling, workflow control
            "human",         # human-in-the-loop / approvals
            "system",        # infra, admin operations
            "other",         # fallback bucket
        ] = Field(default="other", description="Tool category")
    takes_deps: bool = Field(default=False, description="Requires agent context or not")
    parameters: list[ToolParameter] = Field(default_factory=list, description="List of the parameter definitions")
    returns: dict[str,str] = Field(default_factory=dict, description="field_name: description")
    
    is_async: bool = Field(default=False, description="Whether tool execution is async")
    security_level: Literal["standard", "elevated", "admin"] = Field(default="standard", description="Security access level")
    requires_human_approval: bool = Field(default=False)
    can_fail: bool = Field(default=True, description="Whether tool failure is recoverable")
    
    @property
    def parameters_schema(self) -> dict[str,Any]:
        """Gets the JSON schema for the function parameters"""
        properties: dict[str, Any] = {}
        required: list[str] = []
        
        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description
            }
            
            if param.minimum is not None:
                prop["minimum"] = param.minimum
            if param.maximum is not None:
                prop["maximum"] = param.maximum
            if param.enum is not None:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            },
            "returns": self.returns
        }
    
    def validate_parameter_schema(self, parameters: dict[str,Any]) -> Optional[list[str]]:
        """Validate the paramter schema for this tool return None if valid"""    
        schema = self.parameters_schema["parameters"]  # {"type": "object", "properties": {...}}
        
        try:
            validate(instance=parameters, schema=schema)
            return None
        except ValidationError as e:
            # Extract all validation errors
            errors = []
            error = e
            while error:
                errors.append(error.message)
                error = getattr(error, 'context', None)
            return errors       
    @abstractmethod
    def validate_parameters(self, parameters: dict[str,Any]) -> Optional[list[str]]:
        """Validate the paramters for this tool (custom rules) return None if valid"""
        pass

class ToolResult(BaseModel):
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)
    execution_time: float = Field(default=0.0, description="Execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)

#---------------------------------------------------------
#                       Results
#---------------------------------------------------------   
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
        "human_input_received",
        "workflow_completed",
        "retry_required",
        "verification_failed",
        "error"] = Field(..., description="status")
    tool_result: Optional[ToolResult] = Field(default=None, description="Results from tool executions")
    checkpoint: Optional[str] = Field(default=None, description="Name of the checkpoint, if verification")
    error_message: Optional[str] = Field(default=None, description="Error message")
    message: Optional[str] = Field(default=None, description="Message")

#---------------------------------------------------------
#                       State
#---------------------------------------------------------
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

#---------------------------------------------------------
#                       Summary
#---------------------------------------------------------
class Summary(BaseModel):
    summary: str = Field(...,description="Entire workflow summary")
    recommendation: str = Field(..., description="Recommendations for the designer")