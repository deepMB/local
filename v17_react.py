from typing import TypedDict, List, Optional, Any, Dict, Literal, Annotated
from langgraph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
import json
import re
from pydantic import BaseModel, Field
import operator

# Simplified Agent State
class AgentState(TypedDict):
    """Simplified state for the multi-agent system."""
    error_code: int
    sop_content: str
    playbook: Optional[Dict[str, Any]]
    current_step_id: Optional[str]
    execution_status: str
    execution_log: Annotated[List[Dict[str, Any]], operator.add]
    context_data: Dict[str, Any]
    final_output: Dict[str, Any]
    messages: Annotated[List[BaseMessage], operator.add]

# Simplified Playbook Schema
class PlaybookStep(BaseModel):
    """Represents a single step in the playbook"""
    id: str
    name: str
    objective: str
    success_criteria: str
    failure_criteria: str
    next_steps: Dict[str, str]
    description: str

class PlanningAgent:
    """Simplified Planning Agent that creates structured playbooks."""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def __call__(self, state: AgentState) -> AgentState:
        """Create a structured playbook from SOP content."""
        print("🔍 Planning Agent: Analyzing SOP...")
        
        state["execution_status"] = "planning"
        
        system_prompt = f"""You are an expert SOP analyzer. Create a JSON playbook from the given SOP.

RULES:
1. Each step should have clear objectives and success/failure criteria
2. Use "END" for terminal steps
3. Focus on WHAT needs to be achieved, not HOW
4. Create robust error handling paths

Error Code: {state['error_code']}

Return ONLY valid JSON in this format:
{{
    "name": "Playbook name",
    "start_step": "step_1",
    "steps": {{
        "step_1": {{
            "id": "step_1",
            "name": "Step name",
            "objective": "Clear objective",
            "success_criteria": "How to determine success",
            "failure_criteria": "How to determine failure",
            "next_steps": {{"success": "step_2", "failure": "END"}},
            "description": "Detailed description"
        }}
    }}
}}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create playbook for this SOP:\n\n{state['sop_content']}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            playbook = self._extract_and_validate_json(response.content)
            
            state["playbook"] = playbook
            state["current_step_id"] = playbook["start_step"]
            state["execution_status"] = "ready"
            state["messages"].append(HumanMessage(content=f"Playbook created with {len(playbook['steps'])} steps"))
            
            print(f"✅ Created playbook with {len(playbook['steps'])} steps")
            
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            state["execution_status"] = "failed"
            state["final_output"]["error"] = f"Planning failed: {str(e)}"
            
        return state
    
    def _extract_and_validate_json(self, content: str) -> Dict[str, Any]:
        """Extract and validate JSON from LLM response."""
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for JSON-like structure
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                raise ValueError("No valid JSON found in response")
        
        try:
            playbook = json.loads(json_str)
            
            # Validate required fields
            if "steps" not in playbook:
                raise ValueError("Playbook must contain 'steps'")
            if "start_step" not in playbook:
                playbook["start_step"] = list(playbook["steps"].keys())[0]
                
            # Ensure all steps have required fields
            for step_id, step_data in playbook["steps"].items():
                required_fields = ["id", "name", "objective", "success_criteria", "failure_criteria", "next_steps"]
                for field in required_fields:
                    if field not in step_data:
                        if field == "id":
                            step_data[field] = step_id
                        elif field == "next_steps":
                            step_data[field] = {"default": "END"}
                        else:
                            step_data[field] = f"Default {field}"
                            
            return playbook
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

class ExecutionAgent:
    """Simplified Execution Agent using LangGraph's built-in ReAct agent."""
    
    def __init__(self, llm: ChatOpenAI, tools: List[BaseTool]):
        # Create ReAct agent using LangGraph's prebuilt function
        self.react_agent = create_react_agent(llm, tools)
        
    def __call__(self, state: AgentState) -> AgentState:
        """Execute the current step using ReAct agent."""
        if state["execution_status"] != "ready" or not state["current_step_id"]:
            return self._complete_workflow(state)
            
        step_id = state["current_step_id"]
        
        # Handle END step
        if step_id.upper() == "END":
            return self._complete_workflow(state)
            
        # Get current step
        if not state["playbook"] or step_id not in state["playbook"]["steps"]:
            print(f"❌ Step '{step_id}' not found")
            state["execution_status"] = "failed"
            state["final_output"]["error"] = f"Step '{step_id}' not found"
            return state
            
        current_step = state["playbook"]["steps"][step_id]
        print(f"🔧 Executing step: {current_step['name']}")
        
        try:
            # Execute step using ReAct agent
            result = self._execute_step_with_react(current_step, state)
            
            # Log execution
            log_entry = {
                "step_id": step_id,
                "step_name": current_step["name"],
                "objective": current_step["objective"],
                "result": result,
                "status": result["status"]
            }
            state["execution_log"].append(log_entry)
            
            # Update context
            if result.get("context_data"):
                state["context_data"].update(result["context_data"])
            
            # Determine next step
            next_step = self._get_next_step(current_step, result)
            state["current_step_id"] = next_step
            
            print(f"✅ Step completed. Status: {result['status']}, Next: {next_step}")
            
            # Check if workflow should continue
            if next_step == "END" or result.get("critical_failure"):
                return self._complete_workflow(state)
                
        except Exception as e:
            print(f"❌ Execution error: {e}")
            state["execution_status"] = "failed"
            state["final_output"]["error"] = f"Execution failed: {str(e)}"
            
        return state
    
    def _execute_step_with_react(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Execute a step using the ReAct agent."""
        
        # Prepare context for the ReAct agent
        context_info = ""
        if state["context_data"]:
            context_info = f"\nAvailable Context:\n{json.dumps(state['context_data'], indent=2)}"
        
        # Create task description for ReAct agent
        task_message = f"""
WORKFLOW STEP EXECUTION:

Step: {step['name']}
Objective: {step['objective']}
Description: {step.get('description', 'No additional description')}

SUCCESS CRITERIA: {step['success_criteria']}
FAILURE CRITERIA: {step['failure_criteria']}
{context_info}

Your task is to achieve the objective using available tools. 
Evaluate results against the success/failure criteria.
If no suitable tools exist, clearly state this.
Provide a clear assessment of success or failure with reasoning.
"""
        
        # Prepare messages for ReAct agent
        messages = state["messages"] + [HumanMessage(content=task_message)]
        
        try:
            # Invoke ReAct agent
            result = self.react_agent.invoke({"messages": messages})
            
            # Extract the final message from agent
            if result["messages"]:
                agent_response = result["messages"][-1].content
            else:
                agent_response = "No response from agent"
            
            # Analyze the result
            analysis = self._analyze_result(agent_response, step)
            
            # Extract any tool outputs from the conversation
            tool_outputs = self._extract_tool_outputs(result["messages"])
            
            return {
                "status": analysis["status"],
                "message": analysis["message"],
                "agent_response": agent_response,
                "tool_outputs": tool_outputs,
                "context_data": self._extract_context_data(tool_outputs),
                "critical_failure": "NO_SUITABLE_TOOL" in agent_response or "OBJECTIVE_FAILED" in agent_response
            }
            
        except Exception as e:
            return {
                "status": "failure",
                "message": f"ReAct execution failed: {str(e)}",
                "critical_failure": True,
                "error": str(e)
            }
    
    def _analyze_result(self, agent_response: str, step: Dict[str, Any]) -> Dict[str, str]:
        """Analyze agent response to determine success/failure."""
        response_lower = agent_response.lower()
        
        # Check for explicit failure indicators
        failure_indicators = ['failed', 'error', 'unable', 'cannot', 'invalid', 'no_suitable_tool', 'objective_failed']
        success_indicators = ['success', 'completed', 'achieved', 'valid', 'connected', 'verified']
        
        failure_count = sum(1 for indicator in failure_indicators if indicator in response_lower)
        success_count = sum(1 for indicator in success_indicators if indicator in response_lower)
        
        if failure_count > success_count:
            return {
                "status": "failure",
                "message": f"Step failed based on response analysis (failure indicators: {failure_count})"
            }
        elif success_count > 0:
            return {
                "status": "success", 
                "message": f"Step succeeded based on response analysis (success indicators: {success_count})"
            }
        else:
            # Default to success if no clear indicators
            return {
                "status": "success",
                "message": "Step completed (neutral response)"
            }
    
    def _extract_tool_outputs(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract tool outputs from the message chain."""
        tool_outputs = []
        
        for message in messages:
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_outputs.append({
                        "tool": tool_call.get("name", "unknown"),
                        "input": tool_call.get("args", {}),
                    })
            elif hasattr(message, 'content') and isinstance(message.content, str):
                # Look for tool results in content
                if "Tool:" in message.content or "Result:" in message.content:
                    tool_outputs.append({
                        "content": message.content
                    })
                    
        return tool_outputs
    
    def _extract_context_data(self, tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract context data from tool outputs."""
        context = {}
        
        for i, output in enumerate(tool_outputs):
            if "tool" in output:
                context[f"{output['tool']}_result"] = output.get("input", {})
            else:
                context[f"tool_output_{i}"] = output.get("content", "")
                
        return context
    
    def _get_next_step(self, current_step: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Determine the next step based on current step and result."""
        next_steps = current_step.get("next_steps", {})
        status = result.get("status", "unknown")
        
        # Check for critical failure
        if result.get("critical_failure"):
            return "END"
        
        # Look for specific status mapping
        if status in next_steps:
            next_step = next_steps[status]
        elif "default" in next_steps:
            next_step = next_steps["default"]
        else:
            next_step = "END"
            
        # Normalize END variations
        if next_step and next_step.lower() in ['end', 'stop', 'finish', 'complete']:
            return "END"
            
        return next_step
    
    def _complete_workflow(self, state: AgentState) -> AgentState:
        """Complete the workflow and set final status."""
        state["execution_status"] = "completed"
        state["current_step_id"] = None
        
        # Summarize results
        total_steps = len(state["execution_log"])
        successful_steps = sum(1 for log in state["execution_log"] if log["status"] == "success")
        
        state["final_output"].update({
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "completion_status": "success" if successful_steps == total_steps else "partial_success"
        })
        
        print(f"🎉 Workflow completed: {successful_steps}/{total_steps} steps successful")
        return state

def create_workflow(llm: ChatOpenAI, tools: List[BaseTool]) -> StateGraph:
    """Create the simplified multi-agent workflow."""
    
    # Initialize agents
    planning_agent = PlanningAgent(llm)
    execution_agent = ExecutionAgent(llm, tools)
    
    # Create workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_agent)
    workflow.add_node("execution", execution_agent)
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Define routing logic
    def route_after_planning(state: AgentState) -> Literal["execution", "__end__"]:
        """Route after planning based on status."""
        return "execution" if state["execution_status"] == "ready" else "__end__"
    
    def route_after_execution(state: AgentState) -> Literal["execution", "__end__"]:
        """Route after execution based on status."""
        status = state["execution_status"]
        has_next_step = state["current_step_id"] and state["current_step_id"] != "END"
        
        return "execution" if status == "ready" and has_next_step else "__end__"
    
    # Add edges
    workflow.add_conditional_edges(
        "planning", 
        route_after_planning,
        {"execution": "execution", "__end__": END}
    )
    
    workflow.add_conditional_edges(
        "execution",
        route_after_execution, 
        {"execution": "execution", "__end__": END}
    )
    
    return workflow

# Example usage and testing
if __name__ == "__main__":
    from langchain_core.tools import tool
    
    # Mock tools for testing
    @tool
    def validate_data(data_source: str) -> str:
        """Validates data from the specified source."""
        if "invalid" in data_source.lower():
            return f"Data validation FAILED for {data_source}"
        return f"Data validation COMPLETED for {data_source} - 1000 records valid"
    
    @tool
    def check_database_connection(database_name: str) -> str:
        """Checks connection to the specified database."""
        if "test" in database_name.lower():
            return f"Database connection FAILED for {database_name}"
        return f"Database connection VERIFIED for {database_name}"
    
    @tool
    def repair_data(data_source: str, repair_type: str) -> str:
        """Repairs data issues in the specified source."""
        return f"Data repair COMPLETED for {data_source} using {repair_type}"
    
    @tool
    def send_notification(message: str, recipient: str) -> str:
        """Sends a notification message."""
        return f"Notification sent to {recipient}: {message}"
    
    @tool
    def store_in_audit_db(error_code: int) -> str:
        """Stores a record in the audit database."""
        return f"Audit record created for error {error_code}"
    
    # Initialize components
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    tools = [validate_data, check_database_connection, repair_data, send_notification, store_in_audit_db]
    
    # Create and compile workflow
    workflow = create_workflow(llm, tools)
    app = workflow.compile()
    
    # Test SOP
    sample_sop = """
    SOP for Error Code 935 - Data Validation Failure
    
    1. Validate the 'customer_data' source
       - If validation passes: Send success notification and end
       - If validation fails: Proceed to repair
    
    2. Check database connection to 'main_db'
       - If connection fails: Send error notification and end
       - If connection succeeds: Proceed to repair
    
    3. Repair data using 'cleanup' method
       - If repair succeeds: Store audit record and end
       - If repair fails: Send failure notification and end
    """
    
    # Initial state
    initial_state: AgentState = {
        "error_code": 935,
        "sop_content": sample_sop,
        "playbook": None,
        "current_step_id": None,
        "execution_status": "starting",
        "execution_log": [],
        "context_data": {},
        "final_output": {},
        "messages": []
    }
    
    # Run workflow
    try:
        print("🚀 Starting Multi-Agent Workflow with LangGraph ReAct...")
        print("=" * 60)
        
        final_state = app.invoke(initial_state)
        
        print("=" * 60)
        print("📊 WORKFLOW RESULTS:")
        print(f"Status: {final_state['execution_status']}")
        print(f"Steps executed: {len(final_state['execution_log'])}")
        
        if final_state['execution_log']:
            print("\n📋 EXECUTION LOG:")
            for i, log in enumerate(final_state['execution_log'], 1):
                print(f"{i}. {log['step_name']} - {log['status']}")
                print(f"   {log['result']['message']}")
        
        if final_state['final_output']:
            print(f"\n🎯 FINAL RESULTS: {final_state['final_output']}")
            
    except Exception as e:
        print(f"❌ Workflow error: {e}")
        import traceback
        traceback.print_exc()