"""
BEM field solver (MMTL TNT - Mayo Clinic) simulation tool for validation.
"""

import subprocess
from pathlib import Path
import os
from typing import Any
import yaml

from ..core.data_models import ToolParameter, ToolDefinition

PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # Up 3 levels: tools -> src -> root

class BEMSimulator:
    """Boundary Element Method (BEM) simulation execution"""
    
    def __init__(self, config_path: str = "coupled_microstrip_config.yaml") -> None:
        
        full_config_path: Path = PROJECT_ROOT / "config" / config_path
        with open(full_config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.bem_exe: Path = Path(self.config['bem']['executable'])
        self.template: Path = PROJECT_ROOT / Path(self.config['bem']['template'])
        self.result_file: Path = PROJECT_ROOT / Path(self.config['bem']['result_file'])
        self.timeout: int = self.config['bem']['timeout']
    
    def simulate(
        self,
        trace_width_um: float,
        trace_spacing_um: float,
        height_um: float,
        dielectric_constant: float,
        thickness_um: float
    ) -> dict[str, Any]:

        #Run BEM simulation for given geometry
        
        # Update simulation file template
        self._update_template(
            trace_width_um,
            trace_spacing_um,
            height_um,
            dielectric_constant,
            thickness_um
        )
        
        # Run BEM
        self._run_bem()
        
        # Extract results
        z_odd, z_even = self._extract_results()
        
        z_diff: float = 2 * z_odd
        z_comm: float = z_even / 2
        
        return {
            "differential_impedance_ohms": round(z_diff, 2),
            "common_mode_impedance_ohms": round(z_comm, 2),
            "odd_mode_impedance_ohms": round(z_odd, 2),
            "even_mode_impedance_ohms": round(z_even, 2),
            "simulation_method": "2D BEM Field Solver (MMTL TNT - Mayo Clinic)"
        }
    
    def _update_template(self, w: float, s: float, h: float, er: float, t: float):
        """Update ECM#1.xsctn with new parameters."""
        p: float = s + w
        
        with open(self.template, "r") as file:
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
        
        with open(self.template, "w") as file:
            file.writelines(new_lines)
    
    def _run_bem(self) -> None:
        """Execute BEM solver."""
        bem_dir: str = os.path.dirname(self.template)
        ecm_name: str = os.path.splitext(os.path.basename(self.template))[0]
        
        cmd = [self.bem_exe, ecm_name, "10", "10"]
        
        process: subprocess.CompletedProcess[str] = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=bem_dir
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"BEM solver failed with return code {process.returncode}: {process.stderr}")
    
    def _extract_results(self) -> tuple[float, float]:
        """Extract impedances from result file."""
        with open(self.result_file, "r") as file:
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
    
    def validate_parameters(self, parameters: dict[str, Any]) -> None:
        """Validate BEM simulator parameters."""
        pass

BEM_SIMULATOR_TOOL = BEMSimulatorToolDefinition(
    name="simulate_bem",
    description="Simulate coupled microstrip using 2D Boundary Element Method field solver for validation",
    category= "simulation",
    takes_deps=False,
    parameters=[
        ToolParameter(
            name="trace_width_um",
            type="number",
            description="Trace width in micrometers (µm)",
            required=True,
            minimum=50.0,
            maximum=500.0
        ),
        ToolParameter(
            name="trace_spacing_um",
            type="number",
            description="Edge-to-edge spacing in micrometers (µm)",
            required=True,
            minimum=50.0,
            maximum=3000.0
        ),
        ToolParameter(
            name="height_um",
            type="number",
            description="Dielectric height in micrometers (µm)",
            required=True,
            minimum=50.0,
            maximum=500.0
        ),
        ToolParameter(
            name="dielectric_constant",
            type="number",
            description="PCB dielectric constant (Er)",
            required=True,
            minimum=2.0,
            maximum=6.0
        ),
        ToolParameter(
            name="thickness_um",
            type="number",
            description="Copper thickness in micrometers (µm)",
            required=True,
            minimum=5.0,
            maximum=50.0
        )
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