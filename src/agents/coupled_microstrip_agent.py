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
from src.core.data_models import ActionResult, Checkpoint, WorkflowResult

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

# Create list of checkpoints for the workflow
#--------------------------------------------
checkpoints: list[Checkpoint] = [
    Checkpoint(
        name="optimize_coupled_microstrip_geometry_parameters",
        description="Optimize geometric parameters using coupled microstrip optimizer",
        verification_tool_name="simulate_bem" # Match with the name in the ToolDefinition
    ),
]

# Register the associated tools
#--------------------------------------------
tool_registry: ToolRegistry = ToolRegistry()

tool_registry.register_tool(tool_def=cmpo_tool.get_tool_definition(),
                            tool_func=cmpo_tool.get_tool_func())
tool_registry.register_tool(tool_def=bfs_tool.get_tool_definition(),
                            tool_func=bfs_tool.get_tool_func())

# Extend the abstract class to define custom functions
#-----------------------------------------------------
class CoupledStripAgent(PCBAgent):
    def verify_checkpoint(self, checkpoint: Checkpoint, action_result: ActionResult, deps: Any) -> Optional[list[str]]:
        error_messages:list[str] = []
        return error_messages
    
coupled_strip_agent: CoupledStripAgent = CoupledStripAgent(agent_type="Coupled Microstrip Agent",
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