"""
Coupled Microstrip Agent
------------------------
With the main task of running optimisation tools to obtain the required coupled strip geometric parameters
for the given target impedance. Verifications will be done using simulations with a BEM simulator.
"""
from src.core.tool_registry import ToolRegistry
from src.core.pcb_agent import PCBAgent
from src.core.data_models import Checkpoint

from src.tools import coupled_microstrip_parameter_optimizer_tool as cmpo_tool
from src.tools import bem_field_solver_simulator as bfs_tool

# Create list of checkpoints for the workflow