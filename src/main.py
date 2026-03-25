import asyncio
from agents import coupled_microstrip_agent as cmsAgent
from src.core.data_models import WorkflowResult


async def main() -> None:
    query: str = """
    I would like to design a coupled microstrip arrangement to obtain a **target impedance: 90 Ohms**.
    The fixed parametrs for the arrangement are:
    material with a **dielectric constant: 4.5**,
    **height of the arrangement: 250 micro meters**,
    **thickness of the strip: 35 micro meters**,
    and **spacing range in micro meters: (150, 300)**
    
    Use the provided optimisation tools to optimise the geometric parameters for the coupled microstrip arrangement.
    Return the final design after optimisation.
    """
    
    workflow_result: WorkflowResult = await cmsAgent.run_coupled_microstrip_agent(query=query)
    print("\n" + "="*80)
    print("Final workflow results")
    print("="*80)
    print(workflow_result)
    
if __name__ == "__main__":
    asyncio.run(main())