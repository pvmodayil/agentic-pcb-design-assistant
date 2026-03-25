import asyncio
from agents import coupled_microstrip_agent as cmsAgent
from src.core.data_models import WorkflowResult


async def main() -> None:
    query: str = ""
    workflow_result: WorkflowResult = await cmsAgent.run_coupled_microstrip_agent(query=query)

    print(workflow_result)
    
if __name__ == "__main__":
    asyncio.run(main())