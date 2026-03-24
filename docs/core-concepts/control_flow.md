# Overview

## How a workflow runs
 
1. The caller invokes the agent's `run()` method with an initial natural-language query.
2. The agent enters a step loop. At each step, it retrieves relevant context from memory, calls the LLM to decide the next `AgentAction`, and executes it via the `ActionHandler`.
3. If the action is a tool call, the tool is validated and dispatched. Results are stored in state and memory.
4. When the agent reaches a checkpoint verification action, the `VerificationHandler` takes over. For heuristic checkpoints, it extracts parameters from context using an LLM sub-agent, then calls the verifier function. For analytical checkpoints, it runs an LLM review against the verification rule.
5. Verification success advances the workflow. Failure triggers a retry (up to the configured limit) or surfaces the error to the agent for replanning.
6. Human input can be requested at any point. The workflow pauses until a response is provided via the configured `HumanInputProvider`.
7. Once all checkpoints are completed (or the step limit or an error state is reached), a `WorkflowResultBuilder` generates the final `WorkflowResult` using an LLM summary agent. A fallback basic summary is used if the LLM summary fails.
 
---
 
## Example: Coupled Microstrip Agent
 
The `CoupledMicrostripAgent` is a reference implementation that demonstrates the framework's capabilities.
 
**Task:** Given a target differential impedance, find the geometric parameters (trace width and spacing) of a coupled microstrip arrangement that achieves it, and verify the result using a BEM field solver.
 
**Tools registered:**
- `coupled_microstrip_parameter_optimizer` — an ML-based optimizer that proposes geometry given a target impedance
- `simulate_bem` — a boundary element method simulator that computes the actual impedance of a given geometry
 
**Checkpoint:** `optimize_coupled_microstrip_geometry_parameters` uses a heuristic verifier that compares three values: the target impedance, the optimizer's predicted impedance, and the BEM simulation result. It distinguishes three cases:
- BEM converged → success
- Optimizer converged but BEM did not → optimizer is unreliable, failure with diagnostic
- Both failed → complete failure
 
This explicit failure taxonomy ensures that a false positive from the optimizer cannot mask a real discrepancy.
 
---