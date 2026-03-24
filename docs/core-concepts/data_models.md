# Overview

This documentation covers the core data models that power the PCB Design Agentic Workflow Framework. These models define the structure, state, and behavior of the workflow system, enabling consistent communication between components and providing the foundation for the agent's decision-making process.

## Core Data Model Categories

The framework's data models are organized into six key categories:

| Category | Purpose | Key Models |
|----------|---------|------------|
| **Workflow** | Define workflow structure and progression | `Checkpoint`, `ActionType`, `AgentAction` |
| **Tools** | Represent PCB design tools and their execution | `ToolParameter`, `ToolDefinition`, `ToolResult` |
| **State** | Track execution state throughout workflow | `WorkflowState`, `AgentState` |
| **Verification** | Handle checkpoint validation | `ParameterGather`, `VerificationResult` |
| **Results** | Structure final outputs and intermediate results | `FinalResults`, `WorkflowResult`, `ActionResult` |
| **Sub-Agents** | Support multi-agent coordination | `SubAgentConfig`, `PriorityBand` |

## 1. Workflow Models

### Checkpoint Model

The `Checkpoint` model represents a verifiable milestone in the PCB design workflow. Each checkpoint must be validated before proceeding to the next stage.

```python
class Checkpoint(BaseModel):
    name: str
    description: str
    status: Literal["pending", "in_progress", "failed", "completed"] = "pending"
    verification_strategy: Literal["analytical", "heuristics"]
    verification_tool_name: Optional[str] = None
    verification_rule: Optional[str] = None
    verifier_function: Optional[Callable[..., Awaitable["VerificationResult"]]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
```

#### Checkpoint Status Lifecycle

```mermaid
stateDiagram-v2
    [*] --> pending
    pending --> in_progress : Start analysis
    in_progress --> completed : Verification passed
    in_progress --> failed : Verification failed
    failed --> in_progress : Retry
    completed --> [*]
    failed --> [*]
```

*Figure 1: Checkpoint status lifecycle*

#### Verification Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Analytical** | Uses LLM to verify against explicit rules | Subjective validation, complex design rules |
| **Heuristics** | Executes domain-specific Python functions | Quantifiable metrics, precise calculations |

#### Example Checkpoint Definition

```python
power_integrity_cp = Checkpoint(
    name="Power Integrity Verification",
    description="Validate power distribution network stability",
    verification_strategy="heuristics",
    verification_rule="All voltage rails must maintain ripple below 50mV",
    verifier_function=verify_power_integrity,
    metadata={
        "critical_rails": ["VCC_3V3", "VDD_CORE"],
        "max_ripple": 0.05  # 50mV
    }
)
```

### Agent Action Models

The framework uses structured actions to guide the agent's behavior through the workflow.

#### ActionType Enumeration

```python
class ActionType(StrEnum):
    EXECUTE_TOOL = "execute_tool"
    VERIFY_CHECKPOINT = "verify_checkpoint"
    REQUEST_HUMAN_INPUT = "request_human_input"
    UPDATE_CONTEXT = "update_context"
    PROCEED_TO_NEXT = "proceed_to_next"
    RETRY_CHECKPOINT = "retry_checkpoint"
    COMPLETE_WORKFLOW = "complete_workflow"
    ANALYZE = "analyze"
```

#### AgentAction Model

```python
class AgentAction(BaseModel):
    action_type: ActionType
    checkpoint_name: Optional[str] = None
    tool_name: Optional[str] = None
    tool_parameters: dict[str, Any] = Field(default_factory=dict)
    question_for_human: Optional[str] = None
    context_updates: dict[str, Any] = Field(default_factory=dict)
    reasoning: str
    expected_outcome: Optional[str] = None
```

#### Action Flow Diagram

```mermaid
flowchart TD
    A[Analyze Current State] --> B{Decision}
    B -->|Need Tool| C[Execute Tool]
    B -->|Verify| D[Verify Checkpoint]
    B -->|Human Input| E[Request Human Input]
    B -->|Update| F[Update Context]
    B -->|Proceed| G[Proceed to Next]
    B -->|Retry| H[Retry Checkpoint]
    B -->|Complete| I[Complete Workflow]
    
    C --> J[Process Tool Results]
    D --> K{Verification Passed?}
    K -->|Yes| G
    K -->|No| L{Retry Allowed?}
    L -->|Yes| H
    L -->|No| M[Workflow Failed]
    E --> N[Wait for Human Response]
    N --> O[Process Human Input]
    O --> B
    H --> D
    G --> P{All Checkpoints Done?}
    P -->|Yes| I
    P -->|No| B
```

*Figure 2: Agent action flow through the workflow*

## 2. Tool Models

### Tool Parameter Definition

```python
class ToolParameter(BaseModel):
    name: str
    type: Literal["number", "string", "boolean", "object", "array"]
    description: str
    required: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None
```

#### Parameter Validation Flow

```mermaid
sequenceDiagram
    participant Agent
    participant ToolRegistry
    participant ToolDefinition
    
    Agent->>ToolRegistry: Request tool execution
    ToolRegistry->>ToolDefinition: Get parameter schema
    ToolDefinition->>ToolDefinition: Build JSON schema
    ToolDefinition-->>ToolRegistry: Return schema
    ToolRegistry->>ToolDefinition: Validate parameters
    ToolDefinition->>ToolDefinition: Run custom validation
    alt Valid
        ToolDefinition-->>ToolRegistry: No errors
        ToolRegistry->>Tool: Execute with parameters
    else Invalid
        ToolDefinition-->>ToolRegistry: Error messages
        ToolRegistry-->>Agent: Return validation errors
    end
```

*Figure 3: Tool parameter validation flow*

### Tool Definition Model

```python
class ToolDefinition(ABC, BaseModel):
    name: str
    description: str
    category: Literal["io", "network", "retrieval", "calculation", "simulation", 
                     "optimization", "code", "monitoring", "orchestration", 
                     "human", "system", "other"] = "other"
    takes_deps: bool = False
    parameters: list[ToolParameter] = Field(default_factory=list)
    returns: dict[str,str] = Field(default_factory=dict)
    is_async: bool = False
    security_level: Literal["standard", "elevated", "admin"] = "standard"
    requires_human_approval: bool = False
    can_fail: bool = True
```

#### Tool Categories

| Category | Purpose | PCB Design Examples |
|----------|---------|---------------------|
| **calculation** | Mathematical operations | Impedance calculation, thermal analysis |
| **simulation** | Simulation execution | Signal integrity simulation, thermal simulation |
| **optimization** | Optimization algorithms | Component placement optimization |
| **retrieval** | Knowledge lookup | Component database queries |
| **io** | File operations | Gerber file generation, netlist parsing |
| **human** | Human interaction | Approval requests, design reviews |

#### Tool Definition Example

```python
from data_models import ToolDefinition, ToolParameter

class ImpedanceCalculator(ToolDefinition):
    def __init__(self):
        super().__init__(
            name="impedance_calculator",
            description="Calculates trace impedance based on stackup parameters",
            category="calculation",
            parameters=[
                ToolParameter(
                    name="trace_width",
                    type="number",
                    description="Trace width in mm",
                    minimum=0.05,
                    maximum=5.0
                ),
                ToolParameter(
                    name="dielectric_thickness",
                    type="number",
                    description="Dielectric thickness in mm",
                    minimum=0.01,
                    maximum=2.0
                ),
                ToolParameter(
                    name="dielectric_constant",
                    type="number",
                    description="Material Dk value",
                    minimum=2.0,
                    maximum=10.0
                )
            ],
            returns={
                "impedance": "Calculated impedance in ohms",
                "warning": "Any warnings about parameter validity"
            }
        )
    
    def validate_parameters(self, parameters: dict[str, Any]) -> Optional[list[str]]:
        """Custom validation for impedance calculator"""
        errors = []
        if parameters.get("trace_width", 0) < 0.1 and parameters.get("dielectric_thickness", 0) > 1.0:
            errors.append("Very narrow trace with thick dielectric may cause manufacturing issues")
        return errors if errors else None
```

### Tool Result Model

```python
class ToolResult(BaseModel):
    tool_name: str
    success: bool
    result_data: dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

#### Tool Execution Flow

```mermaid
flowchart LR
    A[Agent Requests Tool Execution] --> B{Tool Exists?}
    B -->|Yes| C[Validate Parameters]
    C --> D{Valid?}
    D -->|Yes| E[Execute Tool]
    D -->|No| F[Return Validation Errors]
    E --> G{Success?}
    G -->|Yes| H[Return Results]
    G -->|No| I[Return Error]
    B -->|No| F
```

*Figure 4: Tool execution workflow*

## 3. State Models

### Workflow State Enumeration

```python
class WorkflowState(IntEnum):
    # Informational states
    AWAITING_TOOL_RESULT = 100
    AWAITING_HUMAN = 101
    
    # Success states
    COMPLETED = 200
    PARTIAL_SUCCESS = 201
    TEST_PASSED = 202
    TEST_FAILED = 203
    
    # State transitions
    INITIAL = 300
    ANALYZING = 301
    EXECUTING_TOOL = 302
    TOOL_COMPLETED = 304
    HUMAN_RESPONDED = 306
    TESTING = 307

    # Error states
    ERROR = 500
    VALIDATION_ERROR = 501
    TIMEOUT = 502
    TOOL_ERROR = 503
    AGENT_ERROR = 504
    CHECKPOINT_ERROR = 505
```

#### Workflow State Machine

```mermaid
stateDiagram-v2
    [*] --> INITIAL
    INITIAL --> ANALYZING : Start workflow
    
    ANALYZING --> EXECUTING_TOOL : Execute tool
    EXECUTING_TOOL --> TOOL_COMPLETED : Success
    EXECUTING_TOOL --> TOOL_ERROR : Failure
    
    ANALYZING --> TESTING : Verify checkpoint
    TESTING --> TEST_PASSED : Verification success
    TESTING --> TEST_FAILED : Verification failure
    
    TEST_FAILED --> AWAITING_HUMAN : Request human input
    AWAITING_HUMAN --> HUMAN_RESPONDED : Input received
    HUMAN_RESPONDED --> ANALYZING : Continue workflow
    
    TEST_PASSED --> ANALYZING : Proceed to next
    ANALYZING --> COMPLETED : All checkpoints done
    
    TOOL_ERROR --> AGENT_ERROR
    AGENT_ERROR --> ERROR
    [*] --> ERROR
```

*Figure 5: Complete workflow state machine*

### Agent State Model

```python
class AgentState(BaseModel):
    version: int = 1
    workflow_state: WorkflowState = WorkflowState.INITIAL
    current_checkpoint: Optional[str] = None
    completed_checkpoints: list[str] = Field(default_factory=list)
    pending_checkpoints: list[str] = Field(default_factory=list)
    
    context_data: dict[str, Any] = Field(default_factory=dict)
    tool_results: Optional[ToolResult] = None
    
    needs_human_input: bool = False
    human_question: Optional[str] = None
    human_response: Optional[str] = None
    
    errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
```

#### State Management Flow

```mermaid
flowchart TD
    A[Start Workflow] --> B[Initialize AgentState]
    B --> C{Process Action}
    
    C -->|Tool Execution| D[Update workflow_state to EXECUTING_TOOL]
    D --> E[Store tool parameters in context_data]
    
    C -->|Checkpoint Verification| F[Update workflow_state to TESTING]
    F --> G[Set current_checkpoint]
    
    C -->|Human Input| H[Set needs_human_input = true]
    H --> I[Store human_question]
    
    C -->|Action Success| J[Update completed_checkpoints]
    J --> K{All Checkpoints Done?}
    K -->|Yes| L[Set workflow_state to COMPLETED]
    
    C -->|Action Failure| M[Add to errors list]
    M --> N{Can Retry?}
    N -->|Yes| O[Increment retry_count]
    N -->|No| P[Set workflow_state to ERROR]
```

*Figure 6: Agent state management flow*

## 4. Verification Models

### Parameter Gathering Model

```python
class ParameterGather(BaseModel):
    parameters: dict[str, Any] = Field(..., description="Specified parameters for the verifier function")
```

#### Parameter Extraction Flow

```mermaid
sequenceDiagram
    participant Agent
    participant VerificationHandler
    participant LLM
    
    Agent->>VerificationHandler: verify_checkpoint_with_heuristics()
    VerificationHandler->>LLM: Request parameter extraction
    LLM->>VerificationHandler: Return extracted parameters
    VerificationHandler->>VerificationHandler: Format as ParameterGather
    VerificationHandler->>Verifier: Call with parameters
```

*Figure 7: Parameter extraction flow for heuristic verification*

### Verification Result Model

```python
class VerificationResult(BaseModel):
    success: bool
    notes: Optional[str] = None
    error_messages: Optional[str] = None
```

#### Verification Process

```mermaid
flowchart TD
    A[Start Verification] --> B{Strategy}
    B -->|Analytical| C[LLM Verification]
    B -->|Heuristics| D[Domain Function]
    
    C --> E{Verification Passed?}
    D --> E
    
    E -->|Yes| F[Return success=True]
    E -->|No| G[Return success=False + error messages]
```

*Figure 8: Verification process flow*

## 5. Results Models

### Final Results Model

```python
class FinalResults(BaseModel):
    design_requirements: Optional[dict[str, Any]] = None
    optimization_results: Optional[dict[str, Any]] = None
    performance_metrics: Optional[dict[str, Any]] = None
    component_selections: Optional[dict[str, Any]] = None
    design_summary: Optional[str] = None
    recommendations: Optional[str] = None
```

#### Extending FinalResults

```python
from data_models import FinalResults

class PCBFinalResults(FinalResults):
    """Custom final results structure for PCB design workflows"""
    layer_stackup: dict[str, float] = Field(..., description="Final layer stackup configuration")
    signal_integrity_metrics: dict[str, float] = Field(..., description="Key signal integrity measurements")
    thermal_performance: dict[str, float] = Field(..., description="Thermal analysis results")
    manufacturing_compliance: bool = Field(..., description="Whether design meets manufacturing constraints")
    drc_violations: list[str] = Field(default_factory=list, description="Remaining DRC violations")
```

### Workflow Result Model

```python
class WorkflowResult(BaseModel):
    success: bool
    session_id: str
    workflow_type: str
    final_state: WorkflowState
    
    completed_checkpoints: list[Checkpoint]
    failed_checkpoints: list[Checkpoint]
    
    results: FinalResults
    recommendations: str|None = None
    summary: Optional[str] = None
    
    total_execution_time: float = 0.0
    errors: list[str] = Field(default_factory=list)
```

#### Result Generation Flow

```mermaid
flowchart TD
    A[Workflow Completion] --> B{Success?}
    B -->|Yes| C[Generate Success Results]
    B -->|No| D[Generate Error Results]
    
    C --> E[Compile completed_checkpoints]
    D --> F[Compile failed_checkpoints + errors]
    
    E --> G[Generate design_summary]
    F --> G
    
    G --> H[Generate recommendations]
    H --> I[Build WorkflowResult]
    
    I --> J[Return to User]
```

*Figure 9: Workflow result generation flow*

### Action Result Model

```python
class ActionStatus(StrEnum):
    ANALYZED = "analyzed"
    CONTEXT_UPDATED = "context_updated"
    TOOL_EXECUTED = "tool_executed"
    CHECKPOINT_VERIFIED = "checkpoint_verified"
    HUMAN_INPUT_RECEIVED = "human_input_received"
    PROCEED_TO_NEXT = "proceed_to_next"
    WORKFLOW_COMPLETED = "workflow_completed"
    RETRY_REQUIRED = "retry_required"
    VERIFICATION_FAILED = "verification_failed"
    ERROR = "error"

class ActionResult(BaseModel):
    status: ActionStatus
    tool_result: Optional[ToolResult] = None
    checkpoint: Optional[str] = None
    error_message: Optional[str] = None
    message: Optional[str] = None
```

#### Action Status Transitions

```mermaid
stateDiagram-v2
    [*] --> analyzed
    analyzed --> context_updated : update_context
    analyzed --> tool_executed : execute_tool
    analyzed --> checkpoint_verified : verify_checkpoint
    analyzed --> human_input_received : request_human_input
    analyzed --> proceed_to_next : proceed_to_next
    analyzed --> retry_required : retry_checkpoint
    analyzed --> workflow_completed : complete_workflow
    
    tool_executed --> checkpoint_verified : verify after tool execution
    checkpoint_verified --> proceed_to_next : verification passed
    checkpoint_verified --> retry_required : verification failed
    retry_required --> analyzed : retry with adjustments
```

*Figure 10: Action status transitions*

## 6. Summary Model

```python
class Summary(BaseModel):
    summary: str = Field(..., description="Entire workflow summary")
    recommendation: str = Field(..., description="Recommendations for the designer")
```

#### Summary Generation Process

```mermaid
flowchart TD
    A[Workflow Completion] --> B{LLM Summary?}
    B -->|Yes| C[Generate LLM-based Summary]
    B -->|No| D[Generate Basic Summary]
    
    C --> E[Include completed/failed checkpoints]
    C --> F[Include key metrics]
    C --> G[Include error analysis]
    
    D --> H[Basic success/failure statement]
    D --> I[Simple error listing]
    
    E --> J[Build Summary Object]
    F --> J
    G --> J
    H --> J
    I --> J
    
    J --> K[Return to WorkflowResultBuilder]
```

*Figure 11: Summary generation process*

## 7. Sub-Agent Models

### Priority Band Model

```python
class PriorityBand(IntEnum):
    P0 = 1    # Critical / top priority, reserved
    P1 = 10   # High priority
    P2 = 100  # Medium priority  
    P3 = 1000 # Low priority
    P4 = 10000 # Lowest priority, fallback
```

### Sub-Agent Configuration Model

```python
class SubAgentConfig(BaseModel):
    id: str
    name: str
    type: Literal["worker", "router", "critic"]
    description: str
    capabilities: list[str] = Field(default_factory=list)
    priority: PriorityBand = PriorityBand.P2
    status: Literal["active", "disabled", "experimental"] = "experimental"
    trigger_keywords: list[str] = Field(default_factory=list)
```

#### Sub-Agent Coordination

```mermaid
flowchart TD
    A[Main Agent] --> B{Query Received}
    B --> C[Identify Relevant Sub-Agents]
    
    C --> D[Filter by status=active]
    D --> E[Sort by priority]
    E --> F[Check trigger keywords]
    
    F --> G{Found Matches?}
    G -->|Yes| H[Route to Highest Priority]
    G -->|No| I[Use Default Agent]
    
    H --> J[Process Request]
    I --> J
    
    J --> K[Return Results to Main Agent]
```

*Figure 12: Sub-agent coordination flow*

## Model Relationships

### Complete Data Model Diagram

```mermaid
classDiagram
    class Checkpoint {
        +str name
        +str description
        +str status
        +str verification_strategy
        +str verification_tool_name
        +str verification_rule
        +Callable verifier_function
        +datetime timestamp
        +dict metadata
        +str error_message
        +void mark_completed(dict metadata)
        +void mark_failed(str error, dict metadata)
    }
    
    class AgentAction {
        +ActionType action_type
        +str checkpoint_name
        +str tool_name
        +dict tool_parameters
        +str question_for_human
        +dict context_updates
        +str reasoning
        +str expected_outcome
    }
    
    class ToolDefinition {
        +str name
        +str description
        +str category
        +bool takes_deps
        +list~ToolParameter~ parameters
        +dict returns
        +bool is_async
        +str security_level
        +bool requires_human_approval
        +bool can_fail
        +dict parameters_schema
        +list~str~ validate_parameter_schema(dict params)
        +list~str~ validate_parameters(dict params)
    }
    
    class ToolResult {
        +str tool_name
        +bool success
        +dict result_data
        +str error_message
        +float execution_time
        +dict metadata
    }
    
    class AgentState {
        +int version
        +WorkflowState workflow_state
        +str current_checkpoint
        +list~str~ completed_checkpoints
        +list~str~ pending_checkpoints
        +dict context_data
        +ToolResult tool_results
        +bool needs_human_input
        +str human_question
        +str human_response
        +list~str~ errors
        +int retry_count
        +int max_retries
        +bool can_retry()
        +void increment_retry()
    }
    
    class VerificationResult {
        +bool success
        +str notes
        +str error_messages
    }
    
    class FinalResults {
        +dict design_requirements
        +dict optimization_results
        +dict performance_metrics
        +dict component_selections
        +str design_summary
        +str recommendations
    }
    
    class WorkflowResult {
        +bool success
        +str session_id
        +str workflow_type
        +WorkflowState final_state
        +list~Checkpoint~ completed_checkpoints
        +list~Checkpoint~ failed_checkpoints
        +FinalResults results
        +str recommendations
        +str summary
        +float total_execution_time
        +list~str~ errors
    }
    
    class ActionResult {
        +ActionStatus status
        +ToolResult tool_result
        +str checkpoint
        +str error_message
        +str message
    }
    
    class SubAgentConfig {
        +str id
        +str name
        +str type
        +str description
        +list~str~ capabilities
        +PriorityBand priority
        +str status
        +list~str~ trigger_keywords
    }
    
    AgentState "1" *-- "0..*" Checkpoint : completed/pending
    AgentState "1" *-- "0..1" ToolResult : tool_results
    AgentAction "1" *-- "1" ActionType : action_type
    ActionResult "1" *-- "1" ActionStatus : status
    WorkflowResult "1" *-- "1" FinalResults : results
    WorkflowResult "1" *-- "1" WorkflowState : final_state
    SubAgentConfig "1" *-- "1" PriorityBand : priority
```

*Figure 13: Complete data model relationships*

## Best Practices for Model Usage

### 1. Checkpoint Design
- **Granularity**: Create checkpoints that represent meaningful verification points (not too fine-grained)
- **Clear Rules**: Provide explicit verification rules for analytical verification
- **Metadata**: Use metadata for context-specific parameters needed for verification

### 2. Tool Implementation
- **Validation**: Implement thorough parameter validation in `validate_parameters()`
- **Categories**: Properly categorize tools for better discovery and organization
- **Error Handling**: Set `can_fail=True` for tools where failures are expected and recoverable

### 3. State Management
- **Error Tracking**: Always add meaningful error messages to the `errors` list
- **Retry Strategy**: Configure `max_retries` based on checkpoint criticality
- **Context Data**: Use `context_data` to store intermediate results between steps

### 4. Result Structuring
- **Extensibility**: Extend `FinalResults` for domain-specific PCB design outputs
- **Summarization**: Ensure `design_summary` provides actionable insights
- **Recommendations**: Make recommendations specific and implementable