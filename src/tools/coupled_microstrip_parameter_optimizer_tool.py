"""
ML-based optimization tool to optimize coupled microstrip parameters w.r.t. target Zdiff.
Uses an ONNX neural network model to predict impedance and pymoo genetic algorithm (GA) 
to optimize the parameters based on the predicted value.
"""
import numpy as np
import onnxruntime as ort
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.result import Result
from pymoo.optimize import minimize

from pathlib import Path
from typing import Any, Optional
import yaml

from src.core.data_models import ToolParameter, ToolDefinition
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]  # Up 3 levels: tools -> src -> root

#------------------------------------------
# Internal
#------------------------------------------
def _load_config(config_path: str = "coupled_microstrip_parameter_optimizer.yaml") -> dict[str,Any]:
    """Load and return the the componenets from the config file"""
    full_config_path: Path = PROJECT_ROOT / "config" / "tool_config" / config_path
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract Model-specific settings
    onnx_model_config: dict[str,Any] = {
        'onnx_model_path': PROJECT_ROOT / "models" / config['model']['onnx_path'],
        'norm_min': np.array(config['model']['normalization']['min_values']),
        'norm_max': np.array(config['model']['normalization']['max_values'])
    }
    
    optimization_config: dict[str,float] = {
        'convergence_threshold': config['optimization']['convergence_threshold'],
        'model_accuracy_threshold': config['optimization']['model_accuracy_threshold']
    }
    
    input_parameters: dict[str, dict[str,Any]] = {
        param_name: {
            'type': data.get('type', 'object'),
            'description': data.get('description', ''),
            'required': data.get('required', False),
            'min': data.get('min', None),
            'max': data.get('max', None),    
        }
        for param_name, data in config['input_parameters'].items()
    }
    
    return {'onnx_model_config': onnx_model_config, 'optimization_config': optimization_config, 'input_parameters': input_parameters}

def _normalize(x: np.ndarray, norm_min: np.ndarray, norm_max: np.ndarray) -> np.ndarray:
    """Normalize the parameters to [0,1] range"""
    den = norm_max - norm_min
    den_safe = np.where(den == 0.0, 1.0, den)
    return np.clip((x - norm_min) / den_safe, 0.0, 1.0)

def _predict(x: np.ndarray, 
            norm_min: np.ndarray, 
            norm_max: np.ndarray, 
            model_session: ort.InferenceSession) -> np.ndarray:
    """Predict the impedance with ONNX model"""
    x_norm: np.ndarray = _normalize(x, norm_min, norm_max).astype(np.float32)
    input_name = model_session.get_inputs()[0].name
    output: np.ndarray = np.asarray(model_session.run(None, {input_name: x_norm})[0])
    return output

class ZdiffProblem(Problem):
    def __init__(self, 
            target_zdiff_ohms: float, 
            ga_l: np.ndarray, 
            ga_u: np.ndarray,
            norm_min: np.ndarray, 
            norm_max: np.ndarray,
            model_session: ort.InferenceSession) -> None:
        super().__init__(n_var=5, n_obj=1, xl=ga_l, xu=ga_u)
        self.target: float = target_zdiff_ohms
        self.norm_min: np.ndarray = norm_min
        self.norm_max: np.ndarray = norm_max
        self.model_session: ort.InferenceSession = model_session

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Evaluate objective: minimize (z_diff - target)^2."""
        X: np.ndarray = np.asarray(x, dtype=np.float64)
        Y: np.ndarray = _predict(x=X,
                                norm_min=self.norm_min,
                                norm_max=self.norm_max, 
                                model_session=self.model_session)
        z_odd: float = Y[:, 1].item()
        z_diff: float = 2.0 * z_odd
        out["F"] = (z_diff - self.target) ** 2

class CoupledStripOptimizerToolDefinition(ToolDefinition):
    def _validate_parameter_bounds(self,parameters_from_agent: dict[str, Any]) -> Optional[list[str]]:
        """
        Validate Coupled Strip Parameter Optimiser parameter bounds. 
        Shall return list of error messages if there are any validation errors else None
        """
        
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
    
    def validate_parameters(self, parameters_from_agent: dict[str, Any]) -> Optional[list[str]]:
        """
        Validate Coupled Strip Parameter Optimiser parameters. 
        Shall return list of error messages if there are any validation errors else None
        """
        
        errors: list[str] = []
    
        parameter_bound_errors: list[str] | None = self._validate_parameter_bounds(parameters_from_agent=parameters_from_agent)
        
        if parameter_bound_errors:
            errors.extend(parameter_bound_errors)
        
        return errors if errors else None        

#------------------------------------------
# Public API
#------------------------------------------
def optimize_coupled_strip_parameters(
        target_zdiff_ohms: float,
        
        # Fixed values (None = will be optimized)
        height_um: Optional[float] = None,
        dielectric_constant: Optional[float] = None,
        thickness_um: Optional[float] = None,
        trace_spacing_um: Optional[float] = None,
        
        # Custom ranges (overrides defaults if specified)
        width_range_um: Optional[tuple[float, float]] = None,
        height_range_um: Optional[tuple[float, float]] = None,
        spacing_range_um: Optional[tuple[float, float]] = None,
        thickness_range_um: Optional[tuple[float, float]] = None,
        er_range: Optional[tuple[float, float]] = None,
        
        # Default bounds (used if no custom range and not fixed)
        min_width_um: float = 50.0,
        max_width_um: float = 500.0,
        min_spacing_um: float = 50.0,
        max_spacing_um: float = 3000.0,
        min_height_um: float = 50.0,
        max_height_um: float = 500.0,
        min_thickness_um: float = 5.0,
        max_thickness_um: float = 50.0,
        min_er: float = 2.0,
        max_er: float = 6.0,
        
        pop_size: int = 100,
        n_generations: int = 200) -> dict[str, Any]:
    """
    Optimize coupled microstrip for target differential impedance.
        
    Supports three modes for each parameter:
    1. Fixed: Provide specific value (e.g., height_um=200.0)
    2. Custom range: Provide tuple (e.g., spacing_range_um=(100.0, 500.0))
    3. Default range: Leave both as None (uses min/max bounds)
    """
    # Get ONNX model configs
    config: dict[str,Any] = _load_config()
    onnx_model_config = config['onnx_model_config']
    model_session: ort.InferenceSession = ort.InferenceSession(onnx_model_config['onnx_model_path'])
    norm_min: np.ndarray = onnx_model_config['norm_min']
    norm_max: np.ndarray = onnx_model_config['norm_max']
    
    # Build bounds for [w, h, s, t, er]
    param_configs: list[tuple[str, float | None, tuple[float, float] | None, float, float]] = [
            # (name, fixed_value, custom_range, default_min, default_max)
            ("width", None, width_range_um, min_width_um, max_width_um),
            ("height", height_um, height_range_um, min_height_um, max_height_um),
            ("spacing", trace_spacing_um, spacing_range_um, min_spacing_um, max_spacing_um),
            ("thickness", thickness_um, thickness_range_um, min_thickness_um, max_thickness_um),
            ("dielectric", dielectric_constant, er_range, min_er, max_er)
        ]
    
    ga_l = np.array([])
    ga_u = np.array([])
    param_status: dict[str, Any] = {}
    
    for name, fixed_val, custom_range, default_min, default_max in param_configs:
        if fixed_val is not None:
            # FIXED to specific value
            np.append(ga_l,float(fixed_val))
            np.append(ga_u,float(fixed_val))
            param_status[name] = f"fixed at {fixed_val}"
        elif custom_range is not None:
            # OPTIMIZE in custom range
            np.append(ga_l,float(custom_range[0]))
            np.append(ga_u,float(custom_range[1]))
            param_status[name] = f"optimize in [{custom_range[0]}, {custom_range[1]}]"
        else:
            # OPTIMIZE in default range
            np.append(ga_l,float(default_min))
            np.append(ga_u,float(default_max))
            param_status[name] = f"optimize in [{default_min}, {default_max}]"
    
    ga_l: np.ndarray = np.asarray(ga_l, dtype=np.float64)
    ga_u: np.ndarray = np.asarray(ga_u, dtype=np.float64)
    
    # Run optimization
    problem: ZdiffProblem = ZdiffProblem(target_zdiff_ohms=target_zdiff_ohms,
                                         ga_l=ga_l,
                                         ga_u=ga_u,
                                         norm_min=norm_min,
                                         norm_max=norm_max,
                                         model_session=model_session)
    
    algorithm = GA(pop_size=pop_size, eliminate_duplicates=True)
        
    res: Result = minimize(
        problem,
        algorithm,
        ("n_gen", n_generations),
        seed=42,
        verbose=False
    )

    # Extract results
    x_opt = np.array(res.X, dtype=np.float64).reshape(1, -1)
    y_opt = _predict(x_opt,norm_min,norm_max,model_session)
    
    z_even_opt = float(y_opt[0, 0])
    z_odd_opt = float(y_opt[0, 1])
    z_diff_opt = 2.0 * z_odd_opt
    z_comm_opt = z_even_opt / 2.0
    
    error_percent = abs(z_diff_opt - target_zdiff_ohms) / target_zdiff_ohms * 100
    converged = error_percent < config['optimization_config']['convergence_threshold']
    
    return {
        "optimized_trace_width_um": round(float(x_opt[0, 0]), 2),
        "optimized_height_um": round(float(x_opt[0, 1]), 2),
        "optimized_spacing_um": round(float(x_opt[0, 2]), 2),
        "optimized_thickness_um": round(float(x_opt[0, 3]), 2),
        "optimized_dielectric_constant": round(float(x_opt[0, 4]), 2),
        "achieved_zdiff_ohms": round(z_diff_opt, 2),
        "achieved_zcomm_ohms": round(z_comm_opt, 2),
        "achieved_zodd_ohms": round(z_odd_opt, 2),
        "achieved_zeven_ohms": round(z_even_opt, 2),
        "error_percent": round(error_percent, 2),
        "target_zdiff_ohms": target_zdiff_ohms,
        "converged": converged,
        "method": "Genetic Algorithm + ONNX Surrogate",
        "parameter_status": param_status,
        "generations": n_generations,
        "population_size": pop_size
    }
    

def get_coupled_strip_optimizer_tool_definition() -> CoupledStripOptimizerToolDefinition:
    """
    API to provide the ToolDefinition for the Coupled Strip Optimizer Tool
    """
    
    config: dict[str, Any] = _load_config()
    COUPLED_STRIP_OPTIMIZER_TOOL = CoupledStripOptimizerToolDefinition(
        name="optimize_coupled_strip_parameters",
        description="""Optimize coupled microstrip geometry for 
        target differential impedance using NN surrogate model and GA with range constraint support""",
        category= "optimization",
        takes_deps=False,
        parameters=[
            ToolParameter(
                name=param_name,
                type=data.get('type', 'object'),
                description=data.get('description', ''),
                required=data.get('required', False),
                minimum=data.get('min', None),
                maximum=data.get('max', None)
            )
            for param_name, data in config['input_parameters'].items()
        ],
        
        returns={
        "optimized_trace_width_um": "Optimized trace width",
        "achieved_zdiff_ohms": "Achieved differential impedance",
        "error_percent": "Error percentage from target",
        "converged": "Whether optimization converged",
        "parameter_status": "Status of each parameter (fixed/optimized/range)"
        },
        
        is_async=False,
        security_level="standard",
        requires_human_approval=False,
        can_fail=True
    )
    
    return COUPLED_STRIP_OPTIMIZER_TOOL