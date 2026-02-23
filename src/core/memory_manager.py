from llm_model import get_llm_model
from settings import load_settings

from pydantic_ai import Agent, ModelSettings, ModelMessage, AgentRunResult
from data_models import Summary
#------------------------------------------
# Memory Agent (Think about this more)
#------------------------------------------
class MemoryManager:
    """Memory management agent for relevant context generation"""
    def __init__(self, agent_type: str) -> None:
        system_prompt: str = f"""
        You are an expert PCB design assistant engaged in the '{agent_type}' workflow. You main task is to assist the process by contextualizing 
        relevant information from the entire message log. You must focus only on the most relevant infromation that will help with the query.
        """
        self.message_history: list[ModelMessage] = []
        self.mem_agent = Agent(
            get_llm_model(llm_settings=load_settings(key="mem_llm")),
            model_settings=ModelSettings(temperature=0.3),
            system_prompt=system_prompt
        )
    
    def get_relevant_message_history(self, query: str) -> list[ModelMessage]:
        memory_prompt: str =f"""
        Current query: {query}
        
        From recent conversation, create a COMPACT memory state (200 words max) that:
        1. Summarizes KEY insights/decisions relevant to this query
        2. Notes any ERRORS/feedback from previous steps  
        3. Lists ACTIVE context (current checkpoint, open issues)
        4. Forgets irrelevant details 
        """
        context: list[ModelMessage] = self.mem_agent.run_sync(user_prompt=memory_prompt,
                                               message_history=self.message_history).new_messages()
        
        return context
    
    def _get_relevant_experience(self, query: str) -> str:
        """TODO: Gets the RAG output from the experience layer"""
        return """RELEVANT_EXPERIENCE"""
    
    def get_relevant_context(self, query: str) -> str:
        """TODO: Gets the RAG output from docs"""
        relevant_experience: str = self._get_relevant_experience(query=query)
        relevant_context: str = f"""
        **Experience**: {relevant_experience}
        
        **Knowledge**: 
        """
        return relevant_context
    
    def add_to_message_history(self, new_message: list[ModelMessage]) -> None:
        """Add the new chat message to the message hsitory"""
        self.message_history.extend(new_message)
        
class SummaryAgent:
    def __init__(self, temperature: float = 0.1) -> None:
        self.agent = Agent(
            model=get_llm_model(llm_settings=load_settings(key="mem_llm")),
            model_settings=ModelSettings(temperature=temperature),
            output_type=Summary,
            system_prompt="You are an expert workflow analyst who provides professional summaries and recommendations."
        )
    
    async def generate_summary(self, context: str) -> tuple[str,str]:
        prompt: str = f"""
        Based on the following workflow execution details and failures, 
        please provide a comprehensive executive summary and actionable recommendation:
        
        {context}
        
        Please provide a professional, concise summary that:
        1. Highlights the overall workflow outcome
        2. Summarizes key achievements
        3. Notes any significant issues or failures
        4. Provides a brief assessment of the workflow quality
        
        Please provide 3-5 specific, actionable recommendations:
        1. What should be done differently for future workflows?
        2. What specific improvements can be made to prevent similar failures?
        3. Are there any best practices or guidelines to follow?
        If there are no specific recommendations just say "No Recommendations"
        """
        result: AgentRunResult[Summary] = await self.agent.run(prompt, deps=None, message_history=[])
        
        summary: str = result.output.summary
        recommendations: str = result.output.recommendation
        
        return summary, recommendations