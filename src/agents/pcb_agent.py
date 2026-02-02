from typing import TypeVar, Generic
from abc import ABC, abstractmethod
import uuid
from datetime import datetime

from pydantic_ai import Agent, AgentRunResult, ModelSettings, ModelMessage

from llm_model import get_llm_model
from config.settings import load_settings, LLMSettings
from memory_manager import MemoryManager
from src.agents.data_models import ToolDefinition
from tool_registry import ToolRegistry, ToolFunction
from data_models import AgentState, Checkpoint, AgentAction, WorkflowResult, WorkflowState, ToolResult, ActionResult

import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
        
            
###########################################
# PCB Agent (Abstract agent class)
###########################################
DepsType = TypeVar("DepsType")

class PCBAgent(ABC, Generic[DepsType]):
    """
    Base abstract class for all PCB Agents. 
    This provides the core infrastructure with flexibility 
    for domain specific modifications.
    """
    
    def __init__(
        self, 
        agent_type: str,
        task: str,
        model: str,
        checkpoints: list[str],
        tool_registry: ToolRegistry,
        deps_type: type[DepsType],
        **agent_kwargs) -> None:
        
        self.agent_type: str = agent_type
        self.task: str = task
        self.checkpoints: list[str] = checkpoints
        
        # State management
        self.state = AgentState(
            pending_checkpoints=checkpoints.copy()
        )
        self.checkpoint_objects: dict[str, Checkpoint] = {
            name: Checkpoint(name=name, status="pending")
            for name in checkpoints
        }
        
        # Tool management
        self.tool_registry: ToolRegistry = tool_registry
        
        # Create Agent
        system_prompt: str = self._build_system_prompt()
        
        llm_settings: LLMSettings = load_settings(key="llm")
        self.agent = Agent(
            model=get_llm_model(llm_settings),
            model_settings=ModelSettings(temperature=llm_settings.temperature),
            deps_type=deps_type,
            output_type=AgentAction,
            system_prompt=system_prompt
        )
        
        # Register tools with pydanticai
        self._register_pydanticai_tools()
        
        # Memory manager
        self.memory = MemoryManager(agent_type=agent_type)
        
    def _build_system_prompt(self) -> str:
        """Basic wrapper for the agent specific system prompt"""
        
        tool_descriptions: str = self.tool_registry.get_tool_descriptions()
        
        system_prompt: str = f"""
        You are an expert '{self.agent_type}' agent engaged in the PCB design workflow.
        
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
        - execute_tool: Use a specific tool
        - verify_checkpoint: Verify a checkpoint is complete
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
        """TODO: Iterate over the tools and register it with pydanticai agent"""
        pass
    
    async def run(
        self, 
        initial_query: str,
        deps: DepsType,
        max_steps: int = 50
    ) -> WorkflowResult:
        """
        Main execution loop
        
        This orchestrates the entire workflow
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        logger.info(f"Starting workflow session {session_id}")
        logger.info(f"Initial query: {initial_query}")
        
        current_query = self._get_state_info() + initial_query 
        step_count = 0
        
        try:
            while (step_count < max_steps) and (self.state.retry_count < self.state.max_retries) :
                step_count += 1
                logger.info(f"Step {step_count}: State={self.state.workflow_state.name}")
                
                # Check termination conditions
                if self._should_terminate():
                    break
                
                # Get relevant context
                relevant_message_histoty: list[ModelMessage] = self.memory.get_relevant_message_history(
                    query=current_query,
                )
                
                # Run agent to get next action
                result: AgentRunResult[AgentAction] = await self.agent.run(
                    current_query,
                    deps=deps,
                    message_history=relevant_message_histoty
                )
                
                # Update memory
                self.memory.add_to_message_history(result.new_messages())
                
                # Extract action
                action: AgentAction = result.output
                
                logger.info(f"""
                            Action: {action.action_type}
                            Reason: {action.reasoning}""")
                
                # Execute action
                action_result: ActionResult = self._execute_action(action, deps)
                
                # Update state based on result
                self._update_state_from_action_result(action_result)
                
                # Prepare next query
                current_query: str = self._get_state_info() + self._prepare_next_query(action, action_result)
                
                # Check if human input needed
                if self.state.needs_human_input:
                    # TODO: Handle this in real system (have to integrate with the chat system) 
                    logger.warning("Human input required - workflow paused")
            
            # Generate final result
            end_time: datetime = datetime.now()
            execution_time: float = (end_time - start_time).total_seconds()
            
            return self._generate_workflow_result(
                session_id,
                execution_time,
                success=(self.state.workflow_state == WorkflowState.COMPLETED)
            )
            
        except Exception as e:
            logger.error(f"Workflow error: {e}", exc_info=True)
            self.state.workflow_state = WorkflowState.ERROR
            self.state.errors.append(str(e))
            
            end_time: datetime = datetime.now()
            execution_time: float = (end_time - start_time).total_seconds()
            return self._generate_workflow_result(
                session_id,
                execution_time,
                success=False
            )
    
    def _get_state_info(self) -> str:
        return f"""
        **Current State**:
        - Current checkpoint: {self.state.current_checkpoint or "Not started"}
        - Completed: {', '.join(self.state.completed_checkpoints) or "None"}
        - Pending: {', '.join(self.state.pending_checkpoints)}
        \n           
        """
        
    def _execute_action(
        self, 
        action: AgentAction, 
        deps: DepsType
    ) -> ActionResult:
        """Execute the given action and return results"""
        
        try:
            if action.action_type == "execute_tool":
                return self._execute_tool_action(action, deps)
            
            elif action.action_type == "verify_checkpoint":
                return self._verify_checkpoint_action(action, deps)
            
            elif action.action_type == "request_human_input":
                return self._request_human_input_action(action)
            
            elif action.action_type == "update_context":
                return self._update_context_action(action)
            
            elif action.action_type == "proceed_to_next":
                return self._proceed_to_next_checkpoint(action)
            
            elif action.action_type == "retry_checkpoint":
                return self._retry_checkpoint_action(action)
            
            elif action.action_type == "complete_workflow":
                return self._complete_workflow_action()
            
            else:  # analyze
                return ActionResult(status="analyzed", message=action.reasoning)
        
        except Exception as e:
            logger.error(f"Action execution error: {e}", exc_info=True)
            return ActionResult(status="error", error_message=str(e))
    
    def _execute_tool_action(
        self, 
        action: AgentAction, 
        deps: DepsType
    ) -> ActionResult:
        """Execute a tool and return results"""
        if action.tool_name:
            tool_name: str = action.tool_name
        else:
            return ActionResult(status="error", error_message="No tool name provided")
        
        tool_func: ToolFunction|None = self.tool_registry.get_tool_function(tool_name)
        if not tool_func:
            return ActionResult(status="error", error_message=f"Unknown tool: {tool_name}")
        
        #TODO: Have proper idea on how to pass function parameters
        tool_def: ToolDefinition | None = self.tool_registry.get_tool_definition(tool_name)
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            
            tool_result: ToolResult = tool_func(deps, **action.tool_parameters)
            
            # Store result
            self.state.tool_results = tool_result
            
            return ActionResult(
                status="completed",
                tool_result=tool_result,
            )
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            return ActionResult(status="error", error_message=f"Tool {tool_name} execution failed: {e}")
    
    def _verify_checkpoint_action(
        self, 
        action: AgentAction, 
        deps: DepsType
    ) -> ActionResult:
        """Verify a checkpoint"""
        checkpoint_name: str | None = action.checkpoint_name
        if not checkpoint_name:
            return ActionResult(status="error", error_message="No checkpoint specified")
        
        if checkpoint_name not in self.checkpoint_objects.keys():
            return ActionResult(status="error", error_message=f"Unknown checkpoint: {checkpoint_name}")
        
        checkpoint: Checkpoint = self.checkpoint_objects[checkpoint_name]
        
        # Call domain-specific verification
        is_valid = self.verify_checkpoint(checkpoint, deps)
        
        if is_valid:
            checkpoint.mark_completed()
            self.state.completed_checkpoints.append(checkpoint_name)
            if checkpoint_name in self.state.pending_checkpoints:
                self.state.pending_checkpoints.remove(checkpoint_name)
            
            return ActionResult(status="verified", checkpoint=checkpoint_name)
        else:
            checkpoint.mark_failed("Verification failed")
            return ActionResult(status="verification_failed", checkpoint=checkpoint_name)
    
    def _request_human_input_action(self, action: AgentAction) -> ActionResult:
        """Handle human input request"""
        self.state.needs_human_input = True
        self.state.human_question = action.question_for_human
        self.state.workflow_state = WorkflowState.AWAITING_HUMAN
        
        return ActionResult(status="awaiting_human", message=action.question_for_human)
    
    def _update_context_action(self, action: AgentAction) -> ActionResult:
        """Update workflow context"""
        self.state.context_data.update(action.context_updates)
        return ActionResult(status="context_updated", message=str(action.context_updates))
    
    def _proceed_to_next_checkpoint(self, action: AgentAction) -> ActionResult:
        """Move to next checkpoint"""
        if not self.state.pending_checkpoints:
            self.state.workflow_state = WorkflowState.COMPLETED
            return ActionResult(status="completed", message="All checkpoints completed")
        
        next_checkpoint = self.state.pending_checkpoints.pop(0) # Removes thge first element
        self.state.current_checkpoint = next_checkpoint
        self.checkpoint_objects[next_checkpoint].status = "in_progress"
        
        return ActionResult(status="completed", message=f"Next checkpoint {next_checkpoint}")
    
    def _retry_checkpoint_action(self, action: AgentAction) -> ActionResult:
        """Retry current checkpoint"""
        if not self.state.can_retry():
            return ActionResult(status="error", error_message="Maximum retries exceeded")
        
        self.state.increment_retry()
        
        checkpoint_name = action.checkpoint_name or self.state.current_checkpoint
        if checkpoint_name and checkpoint_name in self.checkpoint_objects.keys():
            self.checkpoint_objects[checkpoint_name].status = "in_progress"
        
        return ActionResult(status="retry", checkpoint=checkpoint_name, message=f"retry number {self.state.retry_count}")
    
    def _complete_workflow_action(self) -> ActionResult:
        """Mark workflow as complete"""
        self.state.workflow_state = WorkflowState.COMPLETED
        return ActionResult(status="completed", message="Workflow completed")
    
    def _should_terminate(self) -> bool:
        """Check if workflow should terminate"""
        terminal_states = {
            WorkflowState.COMPLETED,
            WorkflowState.PARTIAL_SUCCESS,
            WorkflowState.ERROR
        }
        return self.state.workflow_state in terminal_states
    
    def _update_state_from_action_result(
        self, 
        result: ActionResult
    ) -> None:
        """Update agent state based on action result"""
        status = result.status
        
        if status == "error":
            self.state.errors.append(result.error_message if result.error_message else "No error message")
            self.state.workflow_state = WorkflowState.ERROR
        
        elif status == "verification_failed":
            self.state.workflow_state = WorkflowState.TEST_FAILED
        
        elif status == "verified":
            self.state.workflow_state = WorkflowState.TEST_PASSED
        
        elif status == "awaiting_human":
            self.state.workflow_state = WorkflowState.AWAITING_HUMAN
        
        elif status == "completed":
            self.state.workflow_state = WorkflowState.COMPLETED
    
    def _prepare_next_query(
        self, 
        action: AgentAction, 
        result: ActionResult
    ) -> str:
        """Prepare the query for the next agent iteration"""
        status = result.status
        
        if status == "success" and action.action_type == "execute_tool":
            return f"Tool {action.tool_name} executed successfully. Result: {result.tool_result}. What should we do next?"
        
        elif status == "verified":
            return f"Checkpoint {result.checkpoint} verified. Proceed to next step."
        
        elif status == "verification_failed":
            return f"Checkpoint {result.checkpoint} verification failed. Please analyze and suggest fixes."
        
        elif status == "error":
            return f"Error occurred: {result.error_message}. How should we recover?"
        
        else:
            return "Continue with the workflow based on current state."
    
    def _generate_workflow_result(
        self, 
        session_id: str,
        execution_time: float,
        success: bool
    ) -> WorkflowResult:
        #TODO: Final Result should be either another output type from agent or add a section in the current type
        """Generate final workflow result"""
        completed = [
            cp for cp in self.checkpoint_objects.values() 
            if cp.status == "completed"
        ]
        failed = [
            cp for cp in self.checkpoint_objects.values() 
            if cp.status == "failed"
        ]
        
        summary = self._generate_summary(success, completed, failed)
        recommendations = self._generate_recommendations(success, failed)
        
        return WorkflowResult(
            success=success,
            session_id=session_id,
            workflow_type=self.agent_type,
            final_state=self.state.workflow_state,
            completed_checkpoints=completed,
            failed_checkpoints=failed,
            results=self.state.tool_results,
            recommendations=recommendations,
            summary=summary,
            total_execution_time=execution_time,
            errors=self.state.errors
        )
    
    def _generate_summary(
        self, 
        success: bool, 
        completed: list[Checkpoint],
        failed: list[Checkpoint]
    ) -> str:
        """Generate executive summary"""
        if success:
            return f"Workflow completed successfully. {len(completed)} checkpoints completed."
        else:
            return f"Workflow incomplete. {len(completed)} completed, {len(failed)} failed."
    
    def _generate_recommendations(
        self, 
        success: bool,
        failed: list[Checkpoint]
    ) -> list[str]:
        """Generate recommendations"""
        recommendations = []
        
        if not success and failed:
            for cp in failed:
                if cp.error_message:
                    recommendations.append(
                        f"Address failure in '{cp.name}': {cp.error_message}"
                    )
        
        return recommendations
    
    @abstractmethod
    async def verify_checkpoint(
        self, 
        checkpoint: Checkpoint, 
        deps: DepsType
    ) -> bool:
        """
        Verify that a checkpoint has been properly completed
        
        This should be implemented by domain-specific agents
        """
        pass
    
    def provide_human_input(self, response: str) -> None:
        """Provide human response and resume workflow"""
        self.state.human_response = response
        self.state.needs_human_input = False
        self.state.workflow_state = WorkflowState.HUMAN_RESPONDED
        logger.info(f"Human input received: {response[:100]}...")