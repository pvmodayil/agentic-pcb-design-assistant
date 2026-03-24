# Core Concepts
 
### Checkpoints
 
A `Checkpoint` is a named milestone in the workflow. Each checkpoint has a description, a verification strategy, and an optional verification tool or function. The agent must explicitly reach and pass each checkpoint before proceeding. Checkpoints track their own status (`pending`, `in_progress`, `completed`, `failed`) and carry metadata and error messages.
 
### Verification strategies
 
Two strategies are supported, and they can be mixed within a single workflow:
 
- **Heuristics** — a Python async function encodes domain-specific rules. The framework automatically extracts the required parameters from the agent's accumulated context using an LLM parameter-extraction sub-agent, then calls the function and interprets the result. This is appropriate when the pass/fail criterion can be expressed precisely (e.g. impedance error below a threshold, BEM and optimizer results in agreement).
 
- **Analytical** — an LLM verification sub-agent reviews the accumulated context against a natural-language rule. This is appropriate when the criterion involves qualitative judgment or is difficult to encode as a function.
 
### Tools
 
Tools are first-class objects. Each tool is defined as a `ToolDefinition` — a Pydantic model that carries a name, description, category, parameter schema, and return spec. Tool definitions must implement a `validate_parameters` method for domain-specific validation that goes beyond JSON schema. The `ToolRegistry` handles registration, lookup, schema validation, custom validation, and async or sync dispatch.
 
Tool categories include: `io`, `network`, `retrieval`, `calculation`, `simulation`, `optimization`, `code`, `monitoring`, `orchestration`, `human`, and `system`.
 
### Agent actions
 
At each step, the core agent (`PCBAgent`) decides what to do next by emitting a structured `AgentAction`. The possible action types are:
 
- `execute_tool` — call a registered tool with specific parameters
- `verify_checkpoint` — trigger verification of the current checkpoint
- `proceed_to_next` — advance to the next checkpoint
- `retry_checkpoint` — retry after a failure
- `update_context` — store information for downstream steps
- `request_human_input` — pause and ask a human a question
- `complete_workflow` — signal that all checkpoints are done
 
The `ActionHandler` dispatches each action type to the appropriate handler, updates workflow state, and prepares the next query for the agent loop.
 
### Workflow state
 
`WorkflowState` is an integer enum that tracks where the agent is in its lifecycle. States are grouped into informational states (e.g. `AWAITING_TOOL_RESULT`), success states (e.g. `COMPLETED`, `PARTIAL_SUCCESS`), transition states (e.g. `ANALYZING`, `EXECUTING_TOOL`), and error states (e.g. `TIMEOUT`, `TOOL_ERROR`, `CHECKPOINT_ERROR`).
 
### Memory
 
The agent uses a `MemoryManager` to maintain message history across steps. Context retrieval is query-aware — relevant past messages are surfaced for each new agent invocation rather than passing the full history. Memory is flushed and summarized asynchronously at the end of a session.