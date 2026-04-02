# About
 
This project is a **checkpoint-driven agentic workflow framework** for PCB design automation. It provides a structured runtime for LLM-powered agents that execute engineering tasks through verifiable, sequential checkpoints — with built-in tool dispatch, two-mode verification, memory management, and human-in-the-loop support.
 
The framework is domain-agnostic at its core, but is purpose-built for the demands of engineering workflows where correctness matters: results must be verified, not just plausible.
 
---
 
## Why this exists
 
LLM agents are powerful at reasoning and planning, but general-purpose agentic frameworks give no guarantees that intermediate results are actually correct before the workflow proceeds. In engineering contexts — PCB design, impedance matching, signal integrity — a wrong intermediate result that silently propagates is worse than a failure.
 
This framework addresses that by making **checkpoints** the primary unit of progress. An agent cannot advance until the result at each checkpoint has been verified, either by a deterministic heuristic function or by an LLM-based analytical review. Verification failure triggers retry logic or escalates to a human.
 
---
 
## What You Can Build
 
<div class="grid cards" markdown>
 
-   :material-robot-outline: **Custom Agents**
 
    ---
 
    Define agents with specific roles, tool access, and memory — tailored to your PCB design steps.
 
-   :material-tools: **Composable Tool Chains**
 
    ---
 
    Register and wire together tools (EDA integrations, file parsers, validators) via a clean tool registry.
 
-   :material-graph: **Orchestrated Workflows**
 
    ---
 
    Chain agents into multi-step workflows: from schematic review through layout verification to manufacturing checks.
 
-   :material-memory: **Persistent Memory**
 
    ---
 
    Agents can retain context across tasks — tracking design history, component choices, and constraint violations.
 
</div>
 
---
 
## Key design decisions
 
**Checkpoints are the unit of progress, not steps.** The agent loop runs many steps per checkpoint. This means the agent can explore, retry, and replan freely within a checkpoint without the framework losing track of where it is in the overall workflow.
 
**Verification is domain-owned.** The framework provides the mechanism; the domain provides the logic. Heuristic verifier functions are plain Python async functions — no framework-specific base class required. The framework introspects their signatures to gather the right parameters from context automatically.
 
**Tool validation is two-layer.** Every tool call is validated against a JSON schema (via `jsonschema`) and then against a custom `validate_parameters` method. This catches both structural errors (wrong types, missing required fields) and semantic errors (values out of physically meaningful ranges) before any computation runs.
 
**The agent does not see raw tool output.** Tool results are stored in state and selectively surfaced through the memory system. This avoids context bloat and keeps the agent's reasoning focused on what's relevant to the current step.
 
**Failures are typed.** `WorkflowState` error codes distinguish between tool errors, agent errors, checkpoint errors, timeouts, and validation errors. This makes post-hoc debugging tractable and allows the `ActionHandler` to respond differently to different failure modes.

!!! note "Framework vs. Application"
    This is a **workflow framework**, not a finished application. You bring your EDA tools, your design files, and your workflow requirements — the framework provides the agent infrastructure to build on top of.

---

## Project structure
 
```text
config/                             # Configuration files
docs/                               # Zensical docs
outputs/                            # Output files
src/
├── core/                           # Core agent implementation
│   ├── pcb_agent.py                # Base agent class
│   ├── tool_registry.py            # Tool management
│   ├── mcp_server_builder.py       # MCP integration
│   ├── memory_manager.py           # Memory management
│   ├── llm_model.py                # LLM configuration
│   ├── data_models.py              # Pydantic models
│   ├── message_builder.py          # Static methods for building messages
│   ├── settings.py                 # load settings from config file
├── agents/                         # Subagents
├── infrastructure/                 # Infrastructure utilities
├── orchestrator/                   # Workflow orchestration
├── protocols/                      # Protocol definitions
│   ├── input_protocol.py           # Input protocols
├── tools/                          # Tool implementations
└── utils/                          # Utility functions
README.md                           # Main README file
```
 
---

## Contributing

This project is part of the [Information Processing Lab's](https://dt.etit.tu-dortmund.de/en/) agentic AI research initiative.