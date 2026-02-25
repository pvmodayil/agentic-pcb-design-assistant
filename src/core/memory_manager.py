from pydantic_ai.messages import (ModelMessage, 
                                  ModelResponse,
                                  ModelRequest, 
                                  UserPromptPart, 
                                  ThinkingPart)

from pydantic import BaseModel, Field
from loguru import logger
import asyncio

from llm_model import get_llm_model
from settings import load_settings

from pydantic_ai import Agent, ModelSettings, AgentRunResult
from data_models import Summary

#------------------------------------------
# Memory State
#------------------------------------------
class MemoryState(BaseModel):
    """
    Holds the two-layer memory:

    frozen_summary: compressed narrative of everything before the last checkpoint.
                      Built by the LLM; never changes mid-phase.
    live_tail:      raw messages accumulated since the last checkpoint.
                      Cheap to append; no LLM needed.
    _pending_task:  background asyncio.Task compressing the tail into the next
                      frozen_summary (None if idle).
    """
    frozen_summary: str = ""
    live_tail: list[ModelMessage] = Field(default_factory=list)
    _pending_task: asyncio.Task | None = Field(default=None, repr=False)

#------------------------------------------
# Memory Agent (Think about this more)
#------------------------------------------
class MemoryManager:
    """
    Context-engineering strategy
    ----------------------------
    - Between checkpoints  : zero LLM calls; context = frozen_summary + raw tail.
    - At every checkpoint  : one async LLM call compresses tail into frozen_summary.
                             Runs in the background so the main agent is never blocked.
    - get_context()        : always O(1), no LLM, just string concatenation.
    """
    def __init__(self, agent_type: str, max_history_size: int = 50) -> None:
        
        self.TAIL_HARD_LIMIT: int = max_history_size
        self._memory_state: MemoryState = MemoryState()
        system_prompt: str = f"""
        You are a memory compression engine for a PCB design assistant running the
        '{agent_type}' workflow.  Your only job is to produce a COMPACT (<=200 words)
        running summary that preserves:
          - Key decisions and their rationale
          - Errors / warnings and how they were resolved
          - Open issues still needing attention
          - The current workflow phase / checkpoint
        Omit anything not relevant to future steps.
        """
        self.message_history: list[ModelMessage] = []
        self._mem_agent = Agent(
            get_llm_model(llm_settings=load_settings(key="mem_llm")),
            model_settings=ModelSettings(temperature=0.1),
            system_prompt=system_prompt
        )
    
    def _get_relevant_experience(self, query: str) -> str:
        """TODO: Gets the RAG output from the experience layer"""
        return """RELEVANT_EXPERIENCE"""
    
    def _get_relevant_knowledge(self, query: str) -> str:
        """TODO: Gets the RAG output from docs"""
        relevant_experience: str = self._get_relevant_experience(query=query)
        relevant_context: str = f"""
        **Relevant Experience**: {relevant_experience}
        
        **Database Knowledge**: 
        """
        return relevant_context

    def _trigger_compression(self, checkpoint_label: str) -> None:
        """Background compression job. Only one runs at a time."""
        if self._memory_state._pending_task and not self._memory_state._pending_task.done():
            logger.debug("MemoryManager: previous compression still running; skipping trigger.")
            return

        # Snapshot and reset the tail immediately so new messages go into a
        # fresh tail while compression runs in the background.
        live_tail_snapshot: list[ModelMessage] = self._memory_state.live_tail.copy()
        self._memory_state.live_tail.clear()

        task = asyncio.create_task(
            self._compress(live_tail_snapshot, checkpoint_label),
            name=f"memory-compress-{checkpoint_label}",
        )
        self._memory_state._pending_task = task

    async def _compress(self, live_tail_snapshot: list[ModelMessage], checkpoint_label: str) -> None:
        """Merges the tail into frozen_summary via a single background LLM call."""
        prompt: str = f"""
            Checkpoint reached: {checkpoint_label}

            Existing summary (may be empty for the first phase):
            {self._memory_state.frozen_summary or "(none)"}

            Produce an updated compact memory state (<=200 words) from the message history following the system prompt rules.
            """
        try:
            result = await self._mem_agent.run(user_prompt=prompt, message_history=live_tail_snapshot)
            self._memory_state.frozen_summary = self._extract_text(result)
            logger.debug(f"MemoryManager: compression done for checkpoint {checkpoint_label}.")
        except Exception:
            logger.exception("MemoryManager: compression failed; retaining old summary.")
            # Graceful degradation: restore the tail so no messages are lost.
            self._memory_state.live_tail = live_tail_snapshot + self._memory_state.live_tail
    
    def _extract_text(self, result: AgentRunResult) -> str:
        """Pull plain text out of an AgentRunResult regardless of output_type."""
        if hasattr(result, "output") and isinstance(result.output, str):
            return result.output
        for msg in result.new_messages():
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content
        return str(result.output)

    def on_checkpoint(self, checkpoint_label: str) -> None:
        """
        Call this at every workflow checkpoint.

        Triggers a non-blocking background compression of the live tail into
        frozen_summary.  The main agent continues immediately; the result is
        ready by the next checkpoint.
        """
        if not self._memory_state.live_tail:
            logger.debug(f"MemoryManager: checkpoint {checkpoint_label} — tail empty, skipping.")
            return
        self._trigger_compression(checkpoint_label)

    async def flush(self) -> None:
        """
        Await any in-flight compression task.  Call before generating a final
        report or anywhere a guaranteed-fresh summary is needed.
        """
        if self._memory_state._pending_task and not self._memory_state._pending_task.done():
            await self._memory_state._pending_task
                    
    def get_context(self, query: str) -> list[ModelMessage]:
        relevant_knowledge: str = self._get_relevant_knowledge(query)
        context_message: list[ModelMessage] = []

        if self._memory_state.frozen_summary:
            context_message.append(ModelResponse(
                parts=[ThinkingPart(
                    content=f"[**Previous Phases Summary**] {self._memory_state.frozen_summary}."
                )]
            ))

        if self._memory_state.live_tail:
            context_message.extend(self._memory_state.live_tail)

        if relevant_knowledge:
            context_message.append(ModelRequest(
                parts=[UserPromptPart(
                    content=f"[**Relevant Knowledge**] {relevant_knowledge}."
                )]
            ))

        return context_message
    
    def add_to_message_history(self, new_message: list[ModelMessage]) -> None:
        """Add the new chat message to the message hsitory"""
        self._memory_state.live_tail.extend(new_message)
        self.message_history.extend(new_message)
        
        # Safety: force async flush if tail grows very long without a checkpoint.
        if len(self._memory_state.live_tail) >= self.TAIL_HARD_LIMIT:
            logger.warning(
                f"MemoryManager: tail hit hard limit {self.TAIL_HARD_LIMIT}; forcing background compression."
            )
            self._trigger_compression(checkpoint_label="[auto-flush]")
        
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