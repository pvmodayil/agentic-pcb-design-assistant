import asyncio
from src.agents import coupled_microstrip_agent as cmsAgent
from src.core.data_models import WorkflowResult
from loguru import logger
from datetime import datetime
import re

async def main() -> None:
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
    print(workflow_result)
    
if __name__ == "__main__":
    datewise_uid: str =  str(datetime.now())
    # Safe timestamp: replace invalid chars
    datewise_uid = re.sub(r'[: ]', '-', str(datetime.now()))

    # Add handler (returns ID for later removal if needed)
    log_file = f"logs/AgentRunLog_{datewise_uid}.log"
    logger.add(log_file)

    logger.info("AgentRunLog_{datewise_uid}")
    asyncio.run(main())