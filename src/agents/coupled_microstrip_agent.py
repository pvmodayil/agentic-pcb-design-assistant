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

# # Create list of checkpoints for the workflow
# checkpoints: list[Checkpoint] = [
#     Checkpoint(
#         name="define_design_requirements",
#         description="Define the coupled microstrip design requirements including target impedance",
#     ),
#     Checkpoint(
#         name="optimize_geometry_parameters",
#         description="Optimize geometric parameters using coupled microstrip optimizer",
#     ),
#     Checkpoint(
#         name="validate_with_simulation",
#         description="Validate optimized design using BEM field solver simulation",
#     ),
#     Checkpoint(
#         name="verify_impedance_matching",
#         description="Verify that simulated impedance matches target impedance within tolerance",
#     ),
#     Checkpoint(
#         name="finalize_design_specification",
#         description="Finalize and document the complete coupled microstrip design specification",
#     )
# ]