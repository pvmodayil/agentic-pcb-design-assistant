"""
BEM field solver (MMTL TNT - Mayo Clinic) simulation tool for validation.
"""

import subprocess
from pathlib import Path
from typing import Any, Optional
import yaml

from ..core.data_models import ToolParameter, ToolDefinition

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # Up 3 levels: tools -> src -> root

#------------------------------------------
# Internal
#------------------------------------------
def _load_config(config_path: str = "coupled_microstrip_config.yaml") -> dict[str,Any]:
    """Load and return the the componenets from the config file"""
    full_config_path: Path = PROJECT_ROOT / "config" / config_path
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract BEM-specific settings
    bem_config: dict[str,Any] = {
        'executable_path': Path(config['bem']['executable']),
        'template_path': PROJECT_ROOT / config['bem']['template'],
        'resultfile_path': PROJECT_ROOT / config['bem']['resultfile'],
        'timeout': int(config['bem']['timeout']) # value in seconds
    }

    # Extract parameter bounds
    input_parameters: dict[str, dict[str,Any]] = {
        param_name: {
            'type': data.get('type', 'object'),
            'description': data.get('description', ''),
            'required': data.get('required', False),
            'min': data.get('min'),
            'max': data.get('max'),    
        }
        for param_name, data in config['input_parameters'].items()
    }
    
    return {'bem_config': bem_config, 'input_parameters': input_parameters}

def _update_template(template_path: Path, w: float, s: float, h: float, er: float, t: float) -> None:
    """Update template file with new parameters."""
    p: float = s + w
    
    with open(template_path, "r") as file:
        lines: list[str] = file.readlines()
    
    new_lines = []
    thickness_count = 0
    
    for line in lines:
        if "-width" in line:
            new_lines.append(f" -width {w} \\\n")
        elif "-thickness" in line:
            if thickness_count == 1:
                new_lines.append(f" -thickness {h} \\\n")
            else:
                thickness_count += 1
                new_lines.append(line)
        elif "-pitch" in line:
            new_lines.append(f" -pitch {p} \\\n")
        elif "-height" in line:
            new_lines.append(f" -height {t} \\\n")
        elif "-permittivity" in line:
            new_lines.append(f" -permittivity {er} \\\n")
        else:
            new_lines.append(line)
    
    with open(template_path, "w") as file:
        file.writelines(new_lines)

def _run_bem(bem_exe_path: Path, template_path: Path, timeout: int) -> None:
    """Execute BEM solver."""
    bem_dir: Path = template_path.parent
    ecm_name: str = template_path.stem
    
    cmd = [bem_exe_path, ecm_name, "10", "10"]
    
    process: subprocess.CompletedProcess[str] = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=bem_dir
    )
    
    if process.returncode != 0:
        raise RuntimeError(f"BEM solver failed with return code {process.returncode}: {process.stderr}")

def _extract_results(result_file: Path) -> tuple[float, float]:
    """Extract impedances from result file."""
    with open(result_file, "r") as file:
        lines: list[str] = file.readlines()
    
    Z_next = 0
    Z_odd = None
    Z_even = None
    
    for line in lines:
        if "Characteristic Impedance Odd/Even (Ohms):" in line:
            Z_next = 2
        elif Z_next == 2 and "Characteristic Impedance Odd/Even (Ohms):" not in line:
            Z_odd = float(line.split('= ')[1].strip())
            Z_next -= 1
        elif Z_next == 1:
            Z_even = float(line.split('= ')[1].strip())
            Z_next -= 1
    
    if Z_odd is None or Z_even is None:
        raise ValueError("Failed to extract impedances from BEM results")
        
    return Z_odd, Z_even

class BEMSimulatorToolDefinition(ToolDefinition):
    """Concrete implementation of ToolDefinition for BEM simulator."""
    
    def validate_parameters(self, parameters_from_agent: dict[str, Any]) -> Optional[list[str]]:
        """Validate BEM simulator parameters. Shall return list of error messages
        if there are any validation errors else None"""
        
        errors: list[str] = []
    
        # Create a mapping of parameter names to their definitions for easier lookup
        param_definitions: dict[str, ToolParameter] = {param.name: param for param in self.parameters}
        
        # Check for missing required parameters
        for param_def in self.parameters:
            if param_def.required and param_def.name not in parameters_from_agent:
                errors.append(f"Missing **Required Parameter**: {param_def.name}")
        
        # Check further if all the required parameters exist
        if not errors:        
            # Check each provided parameter against its definition
            for param_name, param_value in parameters_from_agent.items():
                if param_name not in param_definitions:
                    errors.append(f"Unknown parameter {param_name}")
                    continue
                    
                param_def: ToolParameter = param_definitions[param_name]
                
                # Validate minimum value constraint
                if param_def.minimum is not None and param_value < param_def.minimum:
                    errors.append(f"Parameter '{param_name}' value {param_value} is less than minimum allowed value {param_def.minimum}")
                
                # Validate maximum value constraint
                if param_def.maximum is not None and param_value > param_def.maximum:
                    errors.append(f"Parameter '{param_name}' value {param_value} is greater than maximum allowed value {param_def.maximum}")
                
                # Validate enum constraints
                if param_def.enum is not None and param_value not in param_def.enum:
                    errors.append(f"Parameter '{param_name}' value '{param_value}' is not in allowed values {param_def.enum}")
        
        return errors if errors else None
    
#------------------------------------------
# Public API
#------------------------------------------
async def simulate_bem(
    trace_width_um: float,
    trace_spacing_um: float,
    height_um: float,
    dielectric_constant: float,
    thickness_um: float,
) -> dict[str, Any]:
    """
    Main simulation function (formerly BEMSimulator.simulate).
    Run BEM simulation for given geometry. Config auto-loads.
    """
    config: dict[str, Any] = _load_config()
    
    bem_config = config['bem_config']
    # Update simulation file template
    _update_template(
        bem_config['template_path'],
        trace_width_um,
        trace_spacing_um,
        height_um,
        dielectric_constant,
        thickness_um
    )
    
    # Run BEM
    _run_bem(bem_config['executable_path'],bem_config['template_path'],bem_config['timeout'])
    
    # Extract results
    z_odd, z_even = _extract_results(bem_config['resultfile_path'])
    
    z_diff: float = 2 * z_odd
    z_comm: float = z_even / 2
    
    return {
        "differential_impedance_ohms": round(z_diff, 2),
        "common_mode_impedance_ohms": round(z_comm, 2),
        "odd_mode_impedance_ohms": round(z_odd, 2),
        "even_mode_impedance_ohms": round(z_even, 2),
        "simulation_method": "2D BEM Field Solver (MMTL TNT - Mayo Clinic)"
    }

def get_bem_simulator_tool_definition() -> BEMSimulatorToolDefinition:
    """
    API to provide the ToolDefinition for the BEMSimulator Tool
    """
    config: dict[str, Any] = _load_config()
    
    BEM_SIMULATOR_TOOL = BEMSimulatorToolDefinition(
        name="simulate_bem",
        description="Simulate coupled microstrip using 2D Boundary Element Method field solver for validation",
        category= "simulation",
        takes_deps=False,
        parameters=[
            ToolParameter(
                name=param_name,
                type=data.get('type', 'object'),
                description=data.get('description', ''),
                required=data.get('required', False),
                minimum=data.get('min'),
                maximum=data.get('max')
            )
            for param_name, data in config['input_parameters'].items()
        ],
        returns={
            "differential_impedance_ohms": "Simulated differential impedance",
            "odd_mode_impedance_ohms": "Odd-mode impedance",
            "even_mode_impedance_ohms": "Even-mode impedance"
        },
        
        is_async=True,
        security_level="standard",
        requires_human_approval=False,
        can_fail=True
    )
    
    return BEM_SIMULATOR_TOOL