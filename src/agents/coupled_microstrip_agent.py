"""
Coupled Microstrip Agent
------------------------
With the main task of running optimisation tools to obtain the required coupled strip geometric parameters
for the given target impedance. Verifications will be done using simulations with a BEM simulator.
"""
import yaml
from pathlib import Path
from typing import Any, Optional
from src.core.tool_registry import ToolRegistry
from src.core.pcb_agent import PCBAgent
from src.core.data_models import  Checkpoint, WorkflowResult, VerificationResult

from src.tools import coupled_microstrip_parameter_optimizer_tool as cmpo_tool
from src.tools import bem_field_solver_simulator as bfs_tool

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # Up 3 levels: agents -> src -> root

#------------------------------------------
# Internal
#------------------------------------------

# Load requirements of the workflow
#--------------------------------------------
def _load_config(config_path: str = "coupled_microstrip.yaml") -> dict[str,Any]:
    """Load and return the the componenets from the config file"""
    full_config_path: Path = PROJECT_ROOT / "config" / "agent_config" /  config_path
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract workflow-specific settings
    workflow: dict[str,Any] = {
        'max_retries': int(config['workflow']['max_retries']),
        'timeout': int(config['workflow']['state_timeout']), # value in seconds
        'auto_retry_on_non_convergence': int(config['workflow']['auto_retry_on_non_convergence'])
    }
    
    # Extract optimization verification settings
    optimization: dict[str,Any] = {
        'convergence_threshold': float(config['workflow']['convergence_threshold']),
        'model_accuracy_threshold': int(config['workflow']['model_accuracy_threshold']),
    }
    
    return {'workflow': workflow, 'optimization': optimization}

# Checkpoint verifier functions (define async)
#--------------------------------------------
async def verify_optimization(target_zdiff: float, optimized_zdiff: float, simulated_zdiff: float) -> VerificationResult:
    workflow_config: dict[str, Any] = _load_config()
    success: bool
    notes: Optional[str]
    error_messages: Optional[str]
    
    # Calculate errors
    sim_error: float = abs(simulated_zdiff - target_zdiff) / target_zdiff * 100
    opt_error: float = abs(optimized_zdiff - target_zdiff) / target_zdiff * 100
    discrepancy: float = abs(optimized_zdiff - simulated_zdiff) / target_zdiff * 100
    
    # Check convergence using threshold
    bem_converged: bool = sim_error < workflow_config["optimization"]["convergence_threshold"]
    ml_converged: bool = opt_error < workflow_config["optimization"]["convergence_threshold"]
    
    if bem_converged:
        # Case 1: BEM converged → SUCCESS
        success = True
        notes = f"""
        Design meets target impedance.
        BEM result: {simulated_zdiff:.2f}Ω (target: {target_zdiff:.2f}Ω, error: {sim_error:.2f}%)
        """
        error_messages = None
        
    
    elif ml_converged and not bem_converged:
        # Case 2: ML says yes, BEM says no → ML/Optimizer unreliable
        success = False
        notes = f"Model simulation dicrepancy in percentage: {round(discrepancy, 2)}"
        error_messages = f"""
            **OPTIMIZER UNRELIABLE**
            ML optimizer predicted convergence, but BEM validation failed."
            ML predicted: {optimized_zdiff:.2f}Ω (error: {opt_error:.2f}%)"
            BEM result: {simulated_zdiff:.2f}Ω (error: {sim_error:.2f}%)"
            The optimizer found a false solution. Using BEM result as ground truth."
        """
        
    else:
        # Case 3: Both failed → Complete failure
        success = False
        notes = None
        error_messages = f"""
        **OPTIMIZATION FAILED**
        Could not achieve target impedance.
        BEM result: {simulated_zdiff:.2f}Ω (target: {target_zdiff:.2f}Ω, error: {sim_error:.2f}%)
        Target not achievable with current constraints
        """
    return VerificationResult(success=success,
                              notes=notes,
                              error_messages=error_messages)
    
# Create list of checkpoints for the workflow
#--------------------------------------------
checkpoints: list[Checkpoint] = [
    Checkpoint(
        name="optimize_coupled_microstrip_geometry_parameters",
        description="Optimize geometric parameters using coupled microstrip optimizer",
        verification_strategy="heuristics",
        verification_tool_name="simulate_bem", # Match with the name in the ToolDefinition
        verifier_function=verify_optimization
    ),
]

# Register the associated tools
#--------------------------------------------
tool_registry: ToolRegistry = ToolRegistry()

tool_registry.register_tool(tool_def=cmpo_tool.get_tool_definition(),
                            tool_func=cmpo_tool.get_tool_func())
tool_registry.register_tool(tool_def=bfs_tool.get_tool_definition(),
                            tool_func=bfs_tool.get_tool_func())

# Define the agent
#-----------------------------------------------------
coupled_strip_agent: PCBAgent = PCBAgent(agent_type="Coupled Microstrip Agent",
                                         task="Optimise the geometric parameters of the coupled microstrip strip arrangement",
                                         list_checkpoints=checkpoints,
                                         tool_registry=tool_registry)
#------------------------------------------
# Public API
#------------------------------------------
async def run_coupled_microstrip_agent(query: str) -> WorkflowResult:
    """Public API to invoke the agent"""
    workflow_result: WorkflowResult = await coupled_strip_agent.run(initial_query=query,)
    
    return workflow_result