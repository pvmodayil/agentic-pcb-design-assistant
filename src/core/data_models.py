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
    description: str = Field(..., description="Description of the checkpoint")
    status: Literal["pending", "in_progress","failed","completed"] = Field(default="pending", description="status of this checkpoint")
    verification_tool_name: Optional[str] = Field(default=None, description="Name of the verification tool")
    verification_rule: str = Field(..., description="Prompt to gverify the checkpoint")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Additional information")
    error_message: Optional[str] = Field(default=None, description="Error message in case of failure")
    
    def mark_completed(self, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mark checkpoint as completed"""
        self.status = "completed"
        self.timestamp = datetime.now()
        if metadata:
            self.metadata.update(metadata) #type:ignore
    
    def mark_failed(self, error: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Mark checkpoint as failed"""
        self.status = "failed"
        self.error_message = error
        self.timestamp = datetime.now()
        if metadata:
            self.metadata.update(metadata) #type:ignore

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
            "optimization",  # optimization tools   
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
    
    def validate_parameter_schema(self, parameters_from_agent: dict[str,Any]) -> Optional[list[str]]:
        """Validate the paramter schema for this tool return None if valid"""    
        schema = self.parameters_schema["parameters"]  # {"type": "object", "properties": {...}}
        
        try:
            validate(instance=parameters_from_agent, schema=schema)
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
    def validate_parameters(self, parameters_from_agent: dict[str,Any]) -> Optional[list[str]]:
        """Validate the paramters for this tool (custom rules) return None if valid"""
        pass

class ToolResult(BaseModel):
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    metadata: dict[str, Any] = Field(default_factory=dict)

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
#                       Results
#---------------------------------------------------------   
class VerificationResult(BaseModel):
    """
    Base class for verification result output
    """
    success: bool = Field(..., description="Verification status")
    error_messages: Optional[str] = Field(default=None, description="Error message when verification fails")
class FinalResults(BaseModel):
    """
    Base class for final results that users can extend.
    This structure will be used by the LLM to generate final results.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # Common fields that might be useful for all workflows
    design_requirements: Optional[dict[str, Any]] = Field(default=None, description="Final design requirements")
    optimization_results: Optional[dict[str, Any]] = Field(default=None, description="Optimization outcomes")
    performance_metrics: Optional[dict[str, Any]] = Field(default=None, description="Performance measurements")
    component_selections: Optional[dict[str, Any]] = Field(default=None, description="Component choices made")
    design_summary: Optional[str] = Field(default=None, description="Summary of final design")
    recommendations: Optional[str] = Field(default=None, description="Recommendations for next steps")
    
class WorkflowResult(BaseModel):
    """Final result of workflow execution"""
    success: bool
    session_id: str
    workflow_type: str
    final_state: WorkflowState
    
    completed_checkpoints: list[Checkpoint]
    failed_checkpoints: list[Checkpoint]
    
    results: FinalResults = Field(..., description="Final results parameters")
    recommendations: str|None = Field(default=None, description="Recommendations from the agent")
    summary: Optional[str] = Field(..., description="Executive summary")
    
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
#                       Summary
#---------------------------------------------------------
class Summary(BaseModel):
    summary: str = Field(...,description="Entire workflow summary")
    recommendation: str = Field(..., description="Recommendations for the designer")
    
#---------------------------------------------------------
#                       Sub Agent
#---------------------------------------------------------
class PriorityBand(IntEnum):
    P0 = 1   # Critical / top priority, reserved
    P1 = 10  # High priority
    P2 = 100 # Medium priority  
    P3 = 1000 # Low priority
    P4 = 10000 # Lowest priority, fallback
    
class SubAgentConfig(BaseModel):
    # Identity
    id: str = Field(..., description="Unique ID for internal routing, e.g. 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'.")
    name: str = Field(..., description="Human readable name.")
    type: Literal["worker","router","critic"] = Field(..., description="Type of the agent")
    
    # Behaviour
    description: str = Field(..., description="What this agent is for and when to use it.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Short capability tags, e.g. ['summarization', 'web_search']."
    )
    
    # Orchestration help
    priority: PriorityBand = Field(default=PriorityBand.P2, description="Lower means more preferred when multiple agents match.")
    status: Literal["active","disabled","experimental"] = Field(default="experimental", description="Status of the agent")
    trigger_keywords: list[str] = Field(default_factory=list, description="If query contains these keywords, this agent is a candidate.")