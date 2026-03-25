# PCB Design Agentic Workflow

## Overview

This framework implements a checkpoint-driven approach where each critical stage of the PCB design process is represented as a verifiable checkpoint. The system intelligently navigates between checkpoints, executes domain-specific tools, verifies results, and handles errors - all while maintaining comprehensive context awareness.

## Core Concepts

### 1. Checkpoint-Driven Workflow

The framework structures PCB design as a sequence of **checkpoints** - critical verification points in the design process. Each checkpoint represents a specific design milestone that must be validated before proceeding.

```mermaid
flowchart LR
    A[Start] --> B[Checkpoint 1: Schematic Review]
    B --> C{Verification}
    C -->|Pass| D[Checkpoint 2: Component Placement]
    C -->|Fail| E[Retry/Correct]
    D --> F{Verification}
    F -->|Pass| G[Checkpoint 3: Routing]
    F -->|Fail| E
    G --> H{Verification}
    H -->|Pass| I[Final Output]
    H -->|Fail| E
    E --> B
    E --> D
    E --> G
```

*Figure 1: Checkpoint verification flow with retry capability*

### 2. Key Components

| Component | Description |
|-----------|-------------|
| **PCBAgent** | Core orchestrator that manages the workflow through checkpoints |
| **Checkpoint** | Represents a verifiable stage in the PCB design process |
| **AgentState** | Tracks current workflow state, completed/pending checkpoints, and errors |
| **ActionHandler** | Processes different action types (tool execution, verification, etc.) |
| **VerificationHandler** | Handles both analytical (LLM-based) and heuristic verification |
| **WorkflowResultBuilder** | Generates final structured results and summaries |

### 3. Verification Strategies

The framework supports two complementary verification approaches:

#### Analytical Verification (LLM-Powered)
```mermaid
sequenceDiagram
    participant Agent as PCBAgent
    participant VH as VerificationHandler
    participant LLM as LLM Model
    
    Agent->>VH: verify_checkpoint_with_llm()
    VH->>LLM: Send verification rule + context
    LLM-->>VH: VerificationResult
    VH->>Agent: Success status or error messages
```

*Figure 2: Analytical verification flow using LLM*

#### Heuristic Verification (Domain-Specific)
```mermaid
sequenceDiagram
    participant Agent as PCBAgent
    participant VH as VerificationHandler
    participant VF as Verifier Function
    
    Agent->>VH: verify_checkpoint_with_heuristics()
    VH->>VF: Extract parameters via LLM
    VF-->>VH: Parameter values
    VH->>VF: Execute verification function
    VF-->>VH: VerificationResult
    VH->>Agent: Success status or error messages
```

*Figure 3: Heuristic verification flow using domain-specific functions*

## Core Architecture

### Agent Context Structure

The `AgentContext` maintains all critical state information throughout the workflow:

```python
@dataclass
class AgentContext:
    state: AgentState                # Current workflow state
    memory: MemoryManager            # Conversation history and context
    tool_registry: ToolRegistry      # Available PCB design tools
    checkpoint_objects: dict[str, Checkpoint]  # All defined checkpoints
    human_input_provider: HumanInputProvider  # Interface for human interaction
```

### Workflow State Machine

The framework implements a comprehensive state machine to track progress:

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

*Figure 4: Workflow state machine diagram*

## Key Classes Deep Dive

### 1. PCBAgent Class

The central orchestrator that manages the entire PCB design workflow.

#### Initialization Parameters
```python
def __init__(
    self, 
    agent_type: str,                # Type of PCB agent (e.g., "RoutingAgent")
    task: str,                      # Specific design task description
    list_checkpoints: list[Checkpoint],  # Ordered list of checkpoints
    tool_registry: ToolRegistry,    # Registered PCB design tools
    max_checkpoint_retries: Optional[int] = None,  # Retry limit per checkpoint
    final_results_type: type[FinalResults] = FinalResults,  # Custom result structure
    deps_type: type[DepsType] = NoDeps,  # Dependency injection type
    temperature: Optional[float] = None,  # LLM temperature setting
    human_input_provider: Optional[HumanInputProvider] = None  # Human interaction interface
)
```

#### Core Workflow Execution
```mermaid
flowchart LR
    A[Start Workflow] --> B{Max Steps Reached?}
    B -->|No| C[Get Relevant Context]
    C --> D[Run Agent for Next Action]
    D --> E[Execute Action via Handler]
    E --> F{Action Successful?}
    F -->|Yes| G[Update State]
    F -->|No| H[Record Error]
    G --> I{Should Terminate?}
    H --> I
    I -->|No| B
    I -->|Yes| J[Generate Final Results]
    J --> K[Return WorkflowResult]
    B -->|Yes| J
```

*Figure 5: Main workflow execution loop*

### 2. Checkpoint Verification System

The framework provides dual verification strategies for robust validation:

#### Analytical Verification
- Uses LLM to verify results against explicit verification rules
- Ideal for subjective or complex validation criteria
- Example verification rule: 
  ```"All high-speed signal traces must maintain impedance within 10% of target value"```

#### Heuristic Verification
- Executes domain-specific Python functions
- Uses LLM to extract required parameters from context
- Example verifier function:
  ```python
  async def verify_power_integrity(voltage_rails: dict, max_ripple: float) -> VerificationResult:
      for rail, ripple in voltage_rails.items():
          if ripple > max_ripple:
              return VerificationResult(success=False, error_messages=[f"{rail} exceeds ripple limit"])
      return VerificationResult(success=True)
  ```

### 3. Action Handling System

The `ActionHandler` processes 8 distinct action types:

| Action Type | Purpose | Key Parameters |
|-------------|---------|----------------|
| `EXECUTE_TOOL` | Run PCB design tools | `tool_name`, `tool_parameters` |
| `VERIFY_CHECKPOINT` | Validate current checkpoint | `checkpoint_name` |
| `REQUEST_HUMAN_INPUT` | Get human guidance | `question_for_human` |
| `UPDATE_CONTEXT` | Modify workflow context | `context_updates` |
| `PROCEED_TO_NEXT` | Move to next checkpoint | - |
| `RETRY_CHECKPOINT` | Retry failed checkpoint | `checkpoint_name` |
| `COMPLETE_WORKFLOW` | Finalize workflow | - |
| `ANALYZE` | Internal planning step | - |

## Workflow Execution Flow

Here's the complete workflow execution sequence:

```mermaid
sequenceDiagram
    participant User
    participant Agent as PCBAgent
    participant AH as ActionHandler
    participant VH as VerificationHandler
    participant TR as ToolRegistry
    
    User->>Agent: Start workflow (run())
    loop For each step
        Agent->>Agent: Prepare state info
        Agent->>Agent: Get relevant context
        Agent->>Agent: Run LLM for next action
        Agent->>AH: Execute action
        alt Tool Execution
            AH->>TR: Execute tool
            TR-->>AH: Tool results
            AH-->>Agent: ActionResult
        else Checkpoint Verification
            AH->>VH: Verify checkpoint
            alt Analytical
                VH->>VH: Prepare verification query
                VH->>LLM: Get verification result
            else Heuristic
                VH->>VH: Extract parameters via LLM
                VH->>Verifier: Execute function
            end
            VH-->>AH: Verification result
            AH-->>Agent: ActionResult
        else Human Input
            AH->>Human: Request input
            Human-->>AH: Provide response
            AH-->>Agent: ActionResult
        end
        Agent->>Agent: Update state
        Agent->>Agent: Prepare next query
    end
    alt Workflow Complete
        Agent->>ResultBuilder: Generate final results
        ResultBuilder-->>Agent: WorkflowResult
        Agent-->>User: Return results
    else Workflow Failed
        Agent->>ResultBuilder: Generate error results
        ResultBuilder-->>Agent: WorkflowResult
        Agent-->>User: Return error results
    end
```

*Figure 6: Complete workflow sequence diagram*

## Error Handling and Recovery

The framework implements robust error handling with multiple recovery strategies:

1. **Checkpoint Retries**
   - Configurable retry limit per checkpoint (`max_checkpoint_retries`)
   - Automatic state reset before retry
   - Progressive error logging

2. **Human Intervention**
   - Seamless transition to human input when stuck
   - Context preservation during human interaction
   - Automatic resumption after input received

3. **Error Classification**
   - Tool errors (invalid parameters, execution failures)
   - Verification failures (design rule violations)
   - Agent errors (planning failures)
   - System errors (framework exceptions)