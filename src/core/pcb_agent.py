from typing import Awaitable, Callable, TypeVar, Generic, Any, Optional
from dataclasses import dataclass

import uuid
from datetime import datetime
import json 

from pydantic_ai import Agent, AgentRunResult, ModelSettings
from pydantic_ai.messages import ModelMessage

from loguru import logger

import llm_model
from settings import load_settings, LLMSettings
from memory_manager import MemoryManager
from tool_registry import ToolRegistry, get_function_parameters
from data_models import (ActionStatus, ActionType, AgentState, 
                        Checkpoint, 
                        AgentAction, 
                        WorkflowResult, 
                        WorkflowState, 
                        ToolResult, 
                        ActionResult, 
                        FinalResults,
                        ParameterGather,
                        VerificationResult)

from message_builder import MessageFactory
from input import HumanInputProvider, ConsoleInputProvider
        
#---------------------------------------------------------
#                    Agent Context
#---------------------------------------------------------
@dataclass
class AgentContext:
    """Contains the main context elements"""
    state: AgentState
    memory: MemoryManager
    tool_registry: ToolRegistry
    checkpoint_objects: dict[str, Checkpoint]
    human_input_provider: HumanInputProvider  
    
#---------------------------------------------------------
#                 Agent Dependency
#---------------------------------------------------------
DepsType = TypeVar("DepsType")

@dataclass
class NoDeps:
    """Must pass NoDeps if there are no dependencies"""
    pass

#---------------------------------------------------------
#               Verification Handler
#---------------------------------------------------------
class VerificationHandler(Generic[DepsType]):
    """Handles Checkpoint verification"""
    def __init__(self, deps_type: type[DepsType] = NoDeps) -> None:
        self.llm_settings: LLMSettings = load_settings(key="llm")
        #-------------------------------------
        # Verification agent
        #-------------------------------------
        self._parameter_agent: Agent[DepsType, ParameterGather] = Agent(
            model=llm_model.get_llm_model(self.llm_settings),
            model_settings=ModelSettings(temperature=0),
            deps_type=deps_type,
            output_type=ParameterGather,
            instructions="""Extract parameters from context exactly as specified.
            Do not infer, estimate, or add keys that were not requested."""
        )
        
        self._verification_agent: Agent[DepsType, VerificationResult] = Agent(
            model=llm_model.get_llm_model(self.llm_settings),
            model_settings=ModelSettings(temperature=0),
            deps_type=deps_type,
            output_type=VerificationResult,
            instructions="Based on the context and the rules mentioned verify the given Checkpoint."
        )
    
    async def verify_checkpoint_with_llm(self, 
                                    checkpoint: Checkpoint,
                                    memory: MemoryManager,
                                    deps: DepsType) -> Optional[str]:
        """
        Prompt the LLM with the given rule to verify the Checkpoint analytically
        """
        verification_query: str = f"""
        You have reached the checkpoint: {checkpoint.name} in the workflow and need to verify that the results obtained so far are accurate.
        You must be precise with your anlysis as accuracy is very important in this workflow.
        
        **Verification Rule**: 
        {checkpoint.verification_rule}
        
        Based on the Verification Rule associated with the checkpoint verify the checkpoint.
        """
        
        # Get relevant context
        relevant_message_history: list[ModelMessage] = memory.get_context(
            query=verification_query,
        )
        
        # Run agent to get next action
        result: AgentRunResult[VerificationResult] = await self._verification_agent.run(
            verification_query,
            deps=deps,
            message_history=relevant_message_history
        )
        
        # Update memory
        memory.add_to_message_history(result.new_messages())
        
        if result.output.success:
            return None
        elif result.output.error_messages:
            return result.output.error_messages
        else:
            return "Verification failed but error messages were not obtained."
    
    async def verify_checkpoint_with_heuristics(self, 
                                                 checkpoint: Checkpoint, 
                                                 memory: MemoryManager,
                                                 deps: DepsType) -> Optional[str]:
        
        if checkpoint.verifier_function is None:
            return "No verifier function is defined with the Checkpoint. Error in definition must ask human"
        else: 
            parameters: dict[str,Any] = await self._gather_parameters(checkpoint, memory, deps)
            if not parameters:
                return "Erro in gathering parameters for the verifier function notify human."
        
        result: VerificationResult = await checkpoint.verifier_function(**parameters)
        
        # Update memory
        if result.notes:
            message: list[ModelMessage] = MessageFactory.build_notes_message(result)
            
            memory.add_to_message_history(message)
     
        if result.success:
            return None
        elif result.error_messages:
            return result.error_messages
        else:
            return "Verification failed but error messages were not obtained."
        
    async def _gather_parameters(self, 
                                 checkpoint: Checkpoint, 
                                 memory: MemoryManager,
                                 deps: DepsType) -> dict[str,Any]:
        """
        Gather parameters by prompting the LLM
        """
        if checkpoint.verifier_function is None:
            return {}
        
        extraction_query: str = f"""
        You are at checkpoint '{checkpoint.name}' and a heuristic verification is required.
    
        Extract ONLY the following parameters from the results accumulated so far.
        
        **Parameters to extract**:
        {get_function_parameters(checkpoint.verifier_function)}
        
        """
        
        # Get relevant context
        relevant_message_history: list[ModelMessage] = memory.get_context(
            query=extraction_query,
        )
        
        # Run agent to get next action
        result: AgentRunResult[ParameterGather] = await self._parameter_agent.run(
            extraction_query,
            deps=deps,
            message_history=relevant_message_history
        )
        
        return result.output.parameters

#---------------------------------------------------------
#               Workflow Result Builder
#---------------------------------------------------------
class WorkflowResultBuilder:
    def __init__(self, final_results_type: type[FinalResults]) -> None:
        self.llm_settings: LLMSettings = load_settings(key="llm")
        #-------------------------------------
        # Result agent
        #-------------------------------------
        self._result_agent: Agent[None, FinalResults] = Agent(
            model=llm_model.get_llm_model(self.llm_settings),
            model_settings=ModelSettings(temperature=0.1),
            output_type=final_results_type,
            instructions="Based on the workflow execution, please generate the final results following the required structure."
        )
    
    async def generate_workflow_result(
        self, 
        session_id: str,
        execution_time: float,
        success: bool,
        context: AgentContext,
        agent_type: str,
    ) -> WorkflowResult:
        """Generate final workflow result"""
        completed: list[Checkpoint] = [
            cp for cp in context.checkpoint_objects.values() 
            if cp.status == "completed"
        ]
        failed: list[Checkpoint] = [
            cp for cp in context.checkpoint_objects.values() 
            if cp.status == "failed"
        ]
        
        final_result: FinalResults = await self._generate_llm_summary_and_results(success=success, 
                                                                                  completed=completed, 
                                                                                  failed=failed, 
                                                                                  memory=context.memory,
                                                                                  agent_type=agent_type,
                                                                                  checkpoint_objects=context.checkpoint_objects)
        
        return WorkflowResult(
            success=success,
            session_id=session_id,
            workflow_type=agent_type,
            final_state=context.state.workflow_state,
            completed_checkpoints=completed,
            failed_checkpoints=failed,
            results=final_result,
            recommendations=final_result.recommendations,
            summary=final_result.design_summary,
            total_execution_time=execution_time,
            errors=context.state.errors
        )
        
    async def _generate_llm_summary_and_results(
        self, 
        success: bool, 
        completed: list[Checkpoint],
        failed: list[Checkpoint],
        memory: MemoryManager,
        agent_type: str,
        checkpoint_objects: dict[str,Checkpoint]
    ) -> FinalResults:
        """LLM based executive summary and recommendations"""
        query: str = f"""
        Workflow Summary Context:
        - Workflow Type: {agent_type}
        - Success Status: {success}
        - Total Checkpoints: {len(checkpoint_objects)}
        - Completed Checkpoints: {len(completed)}
        - Failed Checkpoints: {len(failed)}
        
        Completed Checkpoints:
        {[cp.name for cp in completed]}
        
        Failed Checkpoints:
        {[cp.name for cp in failed]}
        
        Error Messages:
        {self._get_error_messages(success, failed)}
        
        Generate the final results.
        """
        relevant_message_history: list[ModelMessage] = memory.get_context(
                query="Generate the final results",
            )
        
        final_result: AgentRunResult[FinalResults] = await self._result_agent.run(user_prompt=query,
                                                                            message_history=relevant_message_history)
        
        return final_result.output
        
    def _generate_basic_summary(
        self, 
        success: bool, 
        completed: list[Checkpoint],
        failed: list[Checkpoint]
    ) -> tuple[str,str]:
        """Fallback basic summary generator"""
        
        error_messages: list[str] = self._get_error_messages(success, failed)
        if success:
            return f"Workflow completed successfully. {len(completed)} checkpoints completed.", "No Recommendations"
        else:
            return f"Workflow incomplete. {len(completed)} completed, {len(failed)} failed.", "\n".join(error_messages)
    
    def _get_error_messages(
        self, 
        success: bool,
        failed: list[Checkpoint]
    ) -> list[str]:
        """Generate recommendations"""
        error_messages = []
        
        if not success and failed:
            for cp in failed:
                if cp.error_message:
                    error_messages.append(
                        f"Address failure in '{cp.name}': {cp.error_message}"
                    )
        
        return error_messages

#---------------------------------------------------------
#                  Action Handler
#---------------------------------------------------------
# The uniform signature every handler must match
type HandlerFn[DepsType] = Callable[
    [AgentAction, AgentContext, DepsType],
    Awaitable[ActionResult]
]
class ActionHandler(Generic[DepsType]):
    def __init__(self, deps_type: type[DepsType] = NoDeps, ) -> None:
        
        self._action_dispatch: dict[ActionType, HandlerFn] = {
        ActionType.EXECUTE_TOOL:        self._execute_tool_action,
        ActionType.VERIFY_CHECKPOINT:   self._verify_checkpoint_action,
        ActionType.REQUEST_HUMAN_INPUT: self._request_human_input_action,
        ActionType.UPDATE_CONTEXT:      self._update_context_action,
        ActionType.PROCEED_TO_NEXT:     self._proceed_to_next_checkpoint_action,
        ActionType.RETRY_CHECKPOINT:    self._retry_checkpoint_action,
        ActionType.COMPLETE_WORKFLOW:   self._complete_workflow_action,
        }
        #-------------------------------------
        # Verification
        #-------------------------------------
        self._verification_handler: VerificationHandler[DepsType] = VerificationHandler(deps_type=deps_type)
    
    async def execute(
        self,
        action: AgentAction,
        context: AgentContext,
        deps: DepsType,
    ) -> ActionResult:
        """Dispatch the handler from dispatch dictionary"""
        handler: HandlerFn | None = self._action_dispatch.get(action.action_type)
        
        if handler is None:
            return ActionResult(status=ActionStatus.ANALYZED, message=action.reasoning)
        try:
            return await handler(action, context, deps)
        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            return ActionResult(status=ActionStatus.ERROR, error_message=str(e))

    # ------------------------------------------------------------------
    # Handlers — uniform signature: (action, context, deps) -> ActionResult
    # Handlers that don't need a param simply ignore it with _
    # ------------------------------------------------------------------
    async def _execute_tool_action(
        self, 
        action: AgentAction,
        context: AgentContext, 
        deps: DepsType
    ) -> ActionResult:
        """Execute a tool and return results"""
        if action.tool_name:
            tool_name: str = action.tool_name
        else:
            return ActionResult(status=ActionStatus.ERROR, error_message="No tool name provided")
        
        if not action.tool_parameters:
            return ActionResult(status=ActionStatus.ERROR, error_message=f"Tool {tool_name} parameters not provided")
        
        tool_result: ToolResult = await context.tool_registry.handle_tool_call(tool_name=tool_name,
                                                                      tool_parameters=action.tool_parameters)
        context.state.tool_results = tool_result
        
        if tool_result.error_message:
            return ActionResult(status=ActionStatus.ERROR, error_message=tool_result.error_message)
        else:
            return ActionResult(
                status=ActionStatus.TOOL_EXECUTED,
                tool_result=tool_result,
            )
    
    async def _verify_checkpoint_action(
        self, 
        action: AgentAction, 
        context: AgentContext,
        deps: DepsType
    ) -> ActionResult:
        """Verify a checkpoint"""
        checkpoint_name: str | None = action.checkpoint_name
        if not checkpoint_name:
            return ActionResult(status=ActionStatus.ERROR, error_message="No checkpoint specified")
        
        if checkpoint_name not in context.checkpoint_objects.keys():
            return ActionResult(status=ActionStatus.ERROR, error_message=f"Unknown checkpoint: {checkpoint_name}")
        
        checkpoint: Checkpoint = context.checkpoint_objects[checkpoint_name]
        
        # Call domain-specific verification
        if checkpoint.verification_tool_name: # When tool is mentioned with the checkpoint
            if not (action.tool_name == checkpoint.verification_tool_name):
                checkpoint.mark_failed("Verification failed")
                return ActionResult(status=ActionStatus.ERROR, 
                                    checkpoint=checkpoint_name, 
                                    error_message=f"""Given {action.tool_name} is not matching with 
                                    the checkpoint verification tool {checkpoint.verification_tool_name}""")
    
            tool_action_result: ActionResult = await self._execute_tool_action(action, context, deps)
            # Add the tool result to memory if it exists. 
            if tool_action_result.tool_result:
                context.memory.add_to_message_history(
                        MessageFactory.build_tool_return_message(tool_action_result.tool_result) 
                        ) 
        try:
            if checkpoint.verification_strategy == "analytical":    
                error_messages: Optional[str] = await self._verification_handler.verify_checkpoint_with_llm(checkpoint=checkpoint, 
                                                                                                             memory=context.memory,
                                                                                                             deps=deps)
            else: # startegy == "heuristics"
                error_messages: Optional[str] = await self._verification_handler.verify_checkpoint_with_heuristics(checkpoint=checkpoint, 
                                                                                                             memory=context.memory,
                                                                                                             deps=deps)
        except Exception as e:
            logger.exception(f"Exception encountered while verifying Checkpoint: {checkpoint_name}")
            error_messages = f"Verification execution encountered error with message: {e}"
        
        if not error_messages:
            checkpoint.mark_completed()
            context.state.completed_checkpoints.append(checkpoint_name)
            return ActionResult(status=ActionStatus.CHECKPOINT_VERIFIED, checkpoint=checkpoint_name)
        else:
            checkpoint.mark_failed("Verification failed")
            return ActionResult(status=ActionStatus.VERIFICATION_FAILED, checkpoint=checkpoint_name, error_message=error_messages)

    async def _request_human_input_action(self, 
                                    action: AgentAction,
                                    context: AgentContext, 
                                    deps: DepsType) -> ActionResult:
        """Handle human input request"""
        context.state.needs_human_input = True
        context.state.human_question = action.question_for_human
        context.state.workflow_state = WorkflowState.AWAITING_HUMAN
        
        question: str = action.question_for_human if action.question_for_human else "Failed to generate question. Prompt the agent for its query"
        human_response: str = await context.human_input_provider.get_input(question=question)
        
        return ActionResult(status=ActionStatus.HUMAN_INPUT_RECEIVED, message=human_response)
    
    async def _update_context_action(self, 
                               action: AgentAction,
                               context: AgentContext, 
                               deps: DepsType) -> ActionResult:
        """Update workflow context"""
        context.state.context_data.update(action.context_updates)
        return ActionResult(status=ActionStatus.CONTEXT_UPDATED, message=str(action.context_updates))
    
    async def _proceed_to_next_checkpoint_action(self, 
                                                 action: AgentAction,
                                                 context: AgentContext,
                                                 deps: DepsType) -> ActionResult:
        """Move to next checkpoint"""
        if not context.state.pending_checkpoints:
            context.state.workflow_state = WorkflowState.COMPLETED
            return ActionResult(status=ActionStatus.WORKFLOW_COMPLETED, message="All checkpoints completed")
        
        next_checkpoint = context.state.pending_checkpoints.pop(0) # Removes thge first element
        context.state.current_checkpoint = next_checkpoint
        context.checkpoint_objects[next_checkpoint].status = "in_progress"
        
        return ActionResult(status=ActionStatus.PROCEED_TO_NEXT, message=f"Next checkpoint {next_checkpoint}")
    
    async def _retry_checkpoint_action(self, 
                                 action: AgentAction,
                                 context: AgentContext,
                                 deps: DepsType) -> ActionResult:
        """Retry current checkpoint"""
        if not context.state.can_retry():
            return ActionResult(status=ActionStatus.ERROR, error_message="Maximum retries exceeded")
        
        context.state.increment_retry()
        
        checkpoint_name = action.checkpoint_name or context.state.current_checkpoint
        if checkpoint_name and checkpoint_name in context.checkpoint_objects.keys():
            context.checkpoint_objects[checkpoint_name].status = "in_progress"
        
        return ActionResult(status=ActionStatus.RETRY_REQUIRED, checkpoint=checkpoint_name, message=f"retry number {context.state.retry_count}")
    
    async def _complete_workflow_action(self, 
                                  action: AgentAction,
                                  context: AgentContext,
                                  deps: DepsType) -> ActionResult:
        """Mark workflow as complete"""
        context.state.workflow_state = WorkflowState.COMPLETED
        return ActionResult(status=ActionStatus.WORKFLOW_COMPLETED, message="Workflow completed")
         
#------------------------------------------
# PCB Agent (Abstract agent class)
#------------------------------------------
class PCBAgent(Generic[DepsType]):
    """
    PCBAgent
    -------------
    An agentic workflow orchestrator for PCB design based on Checkpoints.
    A predefined list of checkpoints of type Checkpoint is provided, which
    are then individually analyzed and necessary steps/action taken following the
    structured output of AgentAction type. 
    """
    def __init__(
        self, 
        agent_type: str,
        task: str,
        list_checkpoints: list[Checkpoint],
        tool_registry: ToolRegistry,
        max_checkpoint_retries: Optional[int] = None,
        final_results_type: type[FinalResults] = FinalResults,
        deps_type: type[DepsType] = NoDeps,
        temperature: Optional[float] = None,
        human_input_provider: Optional[HumanInputProvider] = None) -> None:
        
        self._agent_type: str = agent_type
        self.task: str = task
        
        #-------------------------------------
        # State Init
        #-------------------------------------
        state = AgentState(
            pending_checkpoints=[checkpoint.name for checkpoint in list_checkpoints],
        )
        if max_checkpoint_retries:
            state.max_retries = max_checkpoint_retries
        
        checkpoint_objects: dict[str, Checkpoint] = {
            checkpoint.name: checkpoint
            for checkpoint in list_checkpoints
        }
        
        #-------------------------------------
        # Context
        #-------------------------------------
        self.context = AgentContext(
            state=state,
            memory=MemoryManager(agent_type=agent_type),
            tool_registry=tool_registry,
            checkpoint_objects=checkpoint_objects,
            human_input_provider=human_input_provider or ConsoleInputProvider()
        )
        
        #-------------------------------------
        # Create main agent
        #-------------------------------------
        system_prompt: str = self._build_system_prompt()
        
        self.llm_settings: LLMSettings = load_settings(key="llm")
        effective_temperature: float = temperature if temperature is not None else self.llm_settings.temperature
        self._agent: Agent[DepsType, AgentAction] = Agent(
            model=llm_model.get_llm_model(self.llm_settings),
            model_settings=ModelSettings(temperature=effective_temperature),
            deps_type=deps_type,
            output_type=AgentAction,
            system_prompt=system_prompt
        )
        #-------------------------------------
        # Action Handler
        #-------------------------------------
        self._action_handler: ActionHandler[DepsType] = ActionHandler(deps_type=deps_type)
        
        #-------------------------------------
        # Result Generator
        #-------------------------------------
        self._workflow_result_builder = WorkflowResultBuilder(final_results_type=final_results_type)
        
    def _build_system_prompt(self) -> str:
        """Basic wrapper for the agent specific system prompt"""
        
        tool_descriptions: str = self.context.tool_registry.get_tool_descriptions()
        
        system_prompt: str = f"""
        You are an expert '{self._agent_type}' agent engaged in the PCB design workflow with the given task/goal.
        
        **Task**: {self.task}
        
        **Available Tools**: {tool_descriptions}
        
        **Your Responsibilities**:
        1. Work through checkpoints systematically
        2. Use tools when needed to gather information or perform calculations
        3. Request human input when facing ambiguity or critical decisions
        4. Verify each checkpoint before proceeding
        5. Provide clear reasoning for all actions

        **Action Types**:
        - analyze: Analyze current situation and plan next steps
        - execute_tool: Use a specific tool from the list of available tools
        - verify_checkpoint: Verify a completed checkpoint
        - request_human_input: Ask human for guidance
        - update_context: Update workflow context with new information
        - proceed_to_next: Move to next checkpoint
        - retry_checkpoint: Retry current checkpoint after fixing issues
        - complete_workflow: Mark entire workflow as complete

        Always respond with an AgentAction specifying:
        - action_type: What to do next
        - reasoning: Why you're taking this action
        - Other relevant fields based on action type
        """
        return system_prompt
    
    def _register_pydanticai_tools(self) -> None:
        """TODO: Iterate over the tools and register it with pydanticai agent
        This however bypasses the fine-grained control for AgentActions as the tool
        execution will be handled in the background before the run result is published.
        """
        pass
    
    #--------------------------
    # Reset
    #--------------------------
    def reset(self) -> None:
        """
        Reset the agent to initial state for a fresh run.
        Clears all state, memory, checkpoint statuses, and tool results.
        """
        # Reset AgentState completely
        self.context.state.version += 1
        self.context.state.workflow_state = WorkflowState.INITIAL
        self.context.state.current_checkpoint = None
        self.context.state.completed_checkpoints = []
        self.context.state.pending_checkpoints = [checkpoint.name for checkpoint in self.context.checkpoint_objects.values()]
        
        self.context.state.context_data = {}
        self.context.state.tool_results = None
        
        self.context.state.needs_human_input = False
        self.context.state.human_question = None
        self.context.state.human_response = None
        
        self.context.state.errors = []
        self.context.state.retry_count = 0
        # max_retries not touched as it is what we got from init
        
        # Reset checkpoint statuses to "pending"
        for checkpoint in self.context.checkpoint_objects.values():
            checkpoint.status = "pending"
            checkpoint.error_message = None
        
        # Clear memory completely
        self.context.memory.clear()

        logger.info(f"{self._agent_type} agent reset complete - ready for fresh run")
        
    #--------------------------
    # Run
    #--------------------------
    async def run(
        self, 
        initial_query: str,
        deps: DepsType = NoDeps(),
        max_steps: int = 50
    ) -> WorkflowResult:
        """
        Main execution loop
        
        This orchestrates the entire workflow
        """
        session_id = str(uuid.uuid4())
        start_time: datetime = datetime.now()
        
        logger.info(f"Starting workflow session {session_id}")
        logger.info(f"Initial query: {initial_query}")
        
        current_query = self._get_workflow_state_info() + initial_query 
        step_count = 0
        
        try:
            while (step_count < max_steps) and (self.context.state.retry_count < self.context.state.max_retries) :
                step_count += 1
                logger.info(f"Step {step_count}: State={self.context.state.workflow_state.name}")
                
                # Check termination conditions
                if self._should_terminate():
                    break
                
                # Get relevant context
                relevant_message_history: list[ModelMessage] = self.context.memory.get_context(
                    query=current_query,
                )
                
                # Run agent to get next action
                result: AgentRunResult[AgentAction] = await self._agent.run(
                    current_query,
                    deps=deps,
                    message_history=relevant_message_history
                )
                
                # Update memory
                self.context.memory.add_to_message_history(result.new_messages())
                
                # Extract action
                action: AgentAction = result.output
                
                logger.info(f"""
                            Action: {action.action_type}
                            Reason: {action.reasoning}""")
                
                # Execute action
                action_result: ActionResult = await self._action_handler.execute(action=action,
                                                                                 context=self.context,
                                                                                 deps=deps)
                
                # Update state based on result
                self._update_state_from_action_result(action_result)
                    
                # Prepare next query
                current_query: str = self._get_workflow_state_info() + self._prepare_next_query(action_result)
            
            # Generate final result
            end_time: datetime = datetime.now()
            execution_time: float = (end_time - start_time).total_seconds()
            
            await self.context.memory.flush() # Wait for pending summarisation tasks
            return await self._workflow_result_builder.generate_workflow_result(
                session_id=session_id,
                execution_time=execution_time,
                success=(self.context.state.workflow_state == WorkflowState.COMPLETED),
                context=self.context,
                agent_type=self._agent_type,
            )
            
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            self.context.state.workflow_state = WorkflowState.ERROR
            self.context.state.errors.append(str(e))
            
            end_time: datetime = datetime.now()
            execution_time: float = (end_time - start_time).total_seconds()
            await self.context.memory.flush() # Wait for pending summarisation tasks
            return await self._workflow_result_builder.generate_workflow_result(
                session_id=session_id,
                execution_time=execution_time,
                success=False,
                context=self.context,
                agent_type=self._agent_type,
            )
    
    #--------------------------
    # Next steps
    #--------------------------
    def _prepare_next_query(self, action_result: ActionResult) -> str:
        """Prepare the query for the next agent iteration"""
        status = action_result.status
        
        if status == ActionStatus.TOOL_EXECUTED:
            if action_result.tool_result:
                self.context.memory.add_to_message_history(
                        MessageFactory.build_tool_return_message(action_result.tool_result)
                        )
                return "Tool executed successfully. Proceed to next steps"
            return "Tool executed successfully. But missing tool results. Retry the tool again."
        
        elif status == ActionStatus.CHECKPOINT_VERIFIED:
            if action_result.checkpoint:
                self.context.memory.on_checkpoint(checkpoint_label=action_result.checkpoint)
            return f"Checkpoint {action_result.checkpoint} verified. Plan and proceed to the next checkpoint."
        
        elif status == ActionStatus.VERIFICATION_FAILED:
            self.context.memory.add_to_message_history(
                    MessageFactory.build_error_messages(action_result)
                    )
            return f"Checkpoint {action_result.checkpoint} verification failed. Please analyze for fixes and retry."
        
        elif status == ActionStatus.ERROR:
            self.context.memory.add_to_message_history(
                    MessageFactory.build_error_messages(action_result)
                    )
            return f"Error occurred: {action_result.error_message}. Address this error and retry?"
        
        elif status == ActionStatus.HUMAN_INPUT_RECEIVED:
            self.context.memory.add_to_message_history(
                    MessageFactory.build_human_input_message(action_result)
                    )
            return "Based on the response from the user proceed to next steps."
        else:
            return "Continue with the workflow based on current state."
        
    #--------------------------
    # State Management
    #--------------------------
    def _get_workflow_state_info(self) -> str:
        """Get state info regarding checkpoints"""
        if self.context.state.current_checkpoint:
            current_cp_metadata: dict[str, Any] | None = self.context.checkpoint_objects[self.context.state.current_checkpoint].metadata
            metadata_str = "No metadata" if current_cp_metadata is None else json.dumps(current_cp_metadata, indent=2, ensure_ascii=False)
            return f"""
            **Current Workflow State**:
            - Current checkpoint: {self.context.state.current_checkpoint}
            - Checkpoint description: {self.context.checkpoint_objects[self.context.state.current_checkpoint].description}
            - Checkpoint metadata: {metadata_str}
            - Completed checkpoints: {', '.join(self.context.state.completed_checkpoints) if self.context.state.completed_checkpoints else "None"}
            - Pending checkpoints: {', '.join(self.context.state.pending_checkpoints) if self.context.state.pending_checkpoints else "None"}
            """
    
        return f"""
        **Current Workflow State**:
        - Current checkpoint: Not started
        - Completed checkpoints: {', '.join(self.context.state.completed_checkpoints) if self.context.state.completed_checkpoints else "None"}
        - Pending checkpoints: {', '.join(self.context.state.pending_checkpoints) if self.context.state.pending_checkpoints else "None"}
        \n           
        """
        
    def _update_state_from_action_result(
        self, 
        action_result: ActionResult
    ) -> None:
        """Update agent state based on action result"""
        status = action_result.status
        
        if status == ActionStatus.ERROR:
            self.context.state.errors.append(action_result.error_message if action_result.error_message else "No error message was provided")
            self.context.state.workflow_state = WorkflowState.ERROR
        
        elif status == ActionStatus.VERIFICATION_FAILED:
            self.context.state.errors.append(action_result.error_message if action_result.error_message else "No error message was provided")
            self.context.state.workflow_state = WorkflowState.TEST_FAILED
        
        elif status == ActionStatus.CHECKPOINT_VERIFIED:
            self.context.state.workflow_state = WorkflowState.TEST_PASSED
        
        elif status == ActionStatus.WORKFLOW_COMPLETED:
            self.context.state.workflow_state = WorkflowState.COMPLETED
    
    #--------------------------
    # Check before resume
    #--------------------------
    def _should_terminate(self) -> bool:
        """Check if workflow should terminate"""
        terminal_states = {
            WorkflowState.COMPLETED,
            WorkflowState.PARTIAL_SUCCESS,
            WorkflowState.ERROR
        }
        return self.context.state.workflow_state in terminal_states