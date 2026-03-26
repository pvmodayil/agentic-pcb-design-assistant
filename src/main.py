import asyncio
from src.agents import coupled_microstrip_agent as cmsAgent
from src.core.data_models import WorkflowResult
from loguru import logger
from datetime import datetime
import re

import json
from pathlib import Path

def save_workflow_result(
    result: WorkflowResult, 
    filename: str = None, 
    directory: str = "outputs",
    format: str = "json"  # "json" or "pickle"
) -> str:
    """Save WorkflowResult to file in JSON or pickle format."""
    # Create directory if it doesn't exist
    Path(directory).mkdir(exist_ok=True)
    
    # Generate timestamp-based filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"workflow_{result.session_id}_{result.workflow_type}_{timestamp}"
        filename = f"{base_name}.{format}"
    
    filepath = Path(directory) / filename
    
    if format == "json":
        # Convert to JSON-serializable dict
        serializable_result = {
            "success": result.success,
            "session_id": result.session_id,
            "workflow_type": result.workflow_type,
            "final_state": result.final_state.dict() if hasattr(result.final_state, 'dict') else str(result.final_state),
            "completed_checkpoints": [cp.model_dump() if hasattr(cp, 'dict') else vars(cp) for cp in result.completed_checkpoints],
            "failed_checkpoints": [cp.model_dump() if hasattr(cp, 'dict') else vars(cp) for cp in result.failed_checkpoints],
            "results": result.results.model_dump() if hasattr(result.results, 'dict') else vars(result.results),
            "recommendations": result.recommendations,
            "summary": result.summary,
            "total_execution_time": result.total_execution_time,
            "errors": result.errors,
            "saved_at": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2, default=str)
    
    else:
        raise ValueError("format must be 'json'")
    
    print(f"✅ WorkflowResult saved to: {filepath}")
    return str(filepath)

async def main(file_name: str) -> None:
    query: str = """
    I want to design a coupled microstrip line to achieve a target differential impedance of 90Ω.

    Use the following fixed parameters:

    Dielectric constant (εr): 4.5

    Substrate height (h): 250µm

    Conductor thickness (t): 35µm

    Spacing range between lines (s): 150µm - 300µm
    Tasks:

    Optimise the geometric parameters for the target differential impedance
    """
    
    workflow_result: WorkflowResult = await cmsAgent.run_coupled_microstrip_agent(query=query)
    print("\n" + "="*80)
    print("Final workflow results")
    print("="*80)
    print(json.dumps(
        workflow_result.model_dump(), 
        indent=2, 
        default=str,
        ensure_ascii=False))
    save_workflow_result(result=workflow_result, filename=file_name)
    
if __name__ == "__main__":
    datewise_uid: str =  str(datetime.now())
    # Safe timestamp: replace invalid chars
    datewise_uid = re.sub(r'[: ]', '-', str(datetime.now()))

    # Add handler (returns ID for later removal if needed)
    file_name: str = f"CoupledStripAgentRun_{datewise_uid}"
    log_file = f"logs/{file_name}.log"
    logger.add(log_file)

    logger.info("AgentRunLog_{datewise_uid}")
    asyncio.run(main(file_name=file_name))