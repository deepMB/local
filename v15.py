from typing import TypedDict, List, Optional, Any, Dict, Literal
from langgraph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import json
import re
from pydantic import BaseModel, Field

# Agent State Definition
class AgentState(TypedDict):
    """
    Defines the shared state for the playbook-driven multi-agent system.
    """
    error_code: int
    sop_content: str
    
    playbook: Optional[Dict[str, Any]]
    """The structured JSON workflow graph generated by the Planning Agent."""

    execution_queue: List[str]
    """A queue of node IDs from the playbook that the Execution Agent needs to process."""
    
    data: Optional[Any]
    """Holds operational data, like the pandas DataFrame for error 935."""

    last_tool_result: Optional[Dict[str, Any]]
    """Stores the output from the last executed tool for conditional evaluation."""

    final_output: Dict[str, Any]
    """A dictionary to accumulate final results and statuses from the workflow."""

    current_step_id: Optional[str]
    """Currently executing step ID."""

    execution_status: str
    """Status of execution: 'planning', 'executing', 'completed', 'failed'."""

    execution_log: List[Dict[str, Any]]
    """Log of all executed steps and their results."""

    context_data: Dict[str, Any]
    """Accumulated context data from all executed steps."""


# Playbook Schema for structured output
class PlaybookStep(BaseModel):
    """Represents a single step in the playbook"""
    id: str = Field(description="Unique identifier for the step")
    name: str = Field(description="Human-readable name of the step")
    action: str = Field(description="The action to be performed")
    objective: str = Field(description="What the step is trying to achieve")
    success_criteria: str = Field(description="How to determine if the step was successful")
    failure_criteria: str = Field(description="How to determine if the step failed")
    next_steps: Dict[str, str] = Field(description="Next steps based on conditions (success, failure, etc.)")
    description: str = Field(description="Detailed description of what this step does")
    context_requirements: Optional[List[str]] = Field(description="What context/data is needed for this step")


class PlanningAgent:
    """
    Enhanced Planning Agent that creates ReAct-friendly playbooks.
    """
    
    def __init__(self, llm: ChatOpenAI, available_tools: List[BaseTool]):
        self.llm = llm
        self.available_tools = available_tools
        self.tool_names = [tool.name for tool in available_tools]
        
    def analyze_sop(self, state: AgentState) -> AgentState:
        """
        Analyzes the SOP content and creates a structured playbook optimized for ReAct execution.
        """
        print("🔍 Planning Agent: Starting SOP analysis...")
        
        sop_content = state["sop_content"]
        error_code = state["error_code"]
        
        # Update execution status
        state["execution_status"] = "planning"
        state["context_data"] = {}
        
        # Create detailed tool descriptions with parameter information
        tool_descriptions = []
        for tool in self.available_tools:
            # Get tool signature information
            tool_desc = f"- {tool.name}: {tool.description}"
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.model_json_schema()
                    if 'properties' in schema:
                        params = []
                        for param_name, param_info in schema['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_desc = param_info.get('description', '')
                            params.append(f"{param_name} ({param_type}): {param_desc}")
                        if params:
                            tool_desc += f"\n    Parameters: {', '.join(params)}"
                except:
                    pass
            tool_descriptions.append(tool_desc)
        
        # Enhanced system prompt for ReAct-optimized planning
        system_prompt = f"""
        You are an expert SOP analyzer that creates ReAct-optimized playbooks for intelligent execution agents.
        
        Your task is to analyze the SOP and create a JSON playbook where each step provides enough context 
        for a ReAct agent to intelligently choose tools and make decisions based on tool outcomes.
        
        AVAILABLE TOOLS:
        {chr(10).join(tool_descriptions)}
        
        CRITICAL DESIGN PRINCIPLES:
        1. Each step should define OBJECTIVES, not specific tools to use
        2. Include clear SUCCESS and FAILURE criteria that a ReAct agent can evaluate
        3. Steps should be tool-agnostic - let the ReAct agent choose the best tool for the objective
        4. Provide rich context about what each step is trying to achieve
        5. Design conditional flows that can handle multiple scenarios
        6. Include parameter guidance for tools that need specific data types
        
        Guidelines for ReAct-optimized playbooks:
        1. Focus on WHAT needs to be achieved, not HOW to achieve it
        2. Provide clear success/failure criteria for each step
        3. Include context requirements (what data/information is needed)
        4. Design steps that allow the ReAct agent to reason about tool selection
        5. Create robust error handling and alternative paths
        6. Each step should be self-contained with clear objectives
        7. When referencing data like error codes, be explicit about the format needed
        
        Error Code Context: {error_code}
        
        IMPORTANT: Return ONLY the JSON playbook, no other text or explanations.
        """
        
        human_prompt = f"""
        Create a ReAct-optimized JSON playbook for this SOP:
        
        {sop_content}
        
        Required JSON structure:
        {{
            "name": "Playbook name",
            "description": "What this playbook accomplishes",
            "start_step": "first_step_id",
            "steps": {{
                "step_id": {{
                    "id": "step_id",
                    "name": "Step name",
                    "action": "achieve_objective|evaluate_condition|notify|end",
                    "objective": "What this step needs to accomplish",
                    "success_criteria": "How to determine success",
                    "failure_criteria": "How to determine failure", 
                    "next_steps": {{"success": "next_id", "failure": "error_id", "default": "END"}},
                    "description": "Detailed description",
                    "context_requirements": ["required_data1", "required_data2"]
                }}
            }}
        }}
        
        RULES:
        1. Use "END" for terminal steps
        2. Focus on objectives, not specific tools
        3. Include rich success/failure criteria
        4. All referenced step IDs must exist
        5. Make it ReAct agent friendly - provide reasoning context
        6. When steps need specific data (like error codes), mention the exact format in the objective
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            playbook_json = self._extract_json_from_response(response.content)
            playbook = self._validate_and_structure_playbook(playbook_json)
            
            state["playbook"] = playbook
            state["execution_queue"] = [playbook["start_step"]] if playbook else []
            state["execution_status"] = "ready_to_execute"
            
            print(f"✅ Planning Agent: Created ReAct-optimized playbook with {len(playbook.get('steps', {}))} steps")
            print(f"📋 Start step: {playbook['start_step']}")
            
        except Exception as e:
            print(f"❌ Planning Agent Error: {str(e)}")
            state["playbook"] = None
            state["execution_queue"] = []
            state["execution_status"] = "failed"
            state["final_output"]["error"] = f"Planning failed: {str(e)}"
        
        return state
    
    def _extract_json_from_response(self, response_content: str) -> Dict[str, Any]:
        """Extracts JSON from the LLM response."""
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response_content, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_content.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError(f"Could not extract valid JSON from response: {str(e)}")
    
    def _validate_and_structure_playbook(self, playbook_json: Dict[str, Any]) -> Dict[str, Any]:
        """Validates the playbook structure."""
        if "steps" not in playbook_json:
            raise ValueError("Playbook must contain 'steps' field")
        
        if "start_step" not in playbook_json:
            playbook_json["start_step"] = list(playbook_json["steps"].keys())[0]
        
        if playbook_json["start_step"] not in playbook_json["steps"]:
            raise ValueError(f"Start step '{playbook_json['start_step']}' not found in steps")
        
        # Ensure each step has required fields
        for step_id, step_data in playbook_json["steps"].items():
            if "id" not in step_data:
                step_data["id"] = step_id
            if "next_steps" not in step_data:
                step_data["next_steps"] = {"default": "END"}
            if "action" not in step_data:
                step_data["action"] = "achieve_objective"
            if "success_criteria" not in step_data:
                step_data["success_criteria"] = "Operation completed successfully"
            if "failure_criteria" not in step_data:
                step_data["failure_criteria"] = "Operation failed or error occurred"
            if "context_requirements" not in step_data:
                step_data["context_requirements"] = []
                
            # Normalize end step references
            if "next_steps" in step_data:
                normalized_next_steps = {}
                for condition, next_step in step_data["next_steps"].items():
                    if isinstance(next_step, str) and next_step.lower() in ['end', 'stop', 'finish', 'complete']:
                        normalized_next_steps[condition] = "END"
                    else:
                        normalized_next_steps[condition] = next_step
                step_data["next_steps"] = normalized_next_steps
        
        return playbook_json


class ExecutionAgent:
    """
    Enhanced Execution Agent that uses ReAct pattern for intelligent tool selection and decision making.
    """
    
    def __init__(self, llm: ChatOpenAI, available_tools: List[BaseTool]):
        self.llm = llm
        self.available_tools = available_tools
        self.tool_dict = {tool.name: tool for tool in available_tools}
        
        # Create detailed tool descriptions with parameter specifications
        tool_descriptions = []
        for tool in available_tools:
            desc = f"{tool.name}: {tool.description}"
            if hasattr(tool, 'args_schema') and tool.args_schema:
                try:
                    schema = tool.args_schema.model_json_schema()
                    if 'properties' in schema:
                        params = []
                        for param_name, param_info in schema['properties'].items():
                            param_type = param_info.get('type', 'unknown')
                            param_desc = param_info.get('description', '')
                            params.append(f"{param_name} ({param_type}): {param_desc}")
                        if params:
                            desc += f" - Parameters: {', '.join(params)}"
                except:
                    pass
            tool_descriptions.append(desc)
        
        # Enhanced ReAct prompt for workflow execution with better parameter handling
        react_template = """You are an intelligent workflow execution agent. You have access to the following tools:

{tools}

Your role is to execute workflow steps by:
1. Understanding the objective of each step
2. Choosing the most appropriate tool(s) to achieve the objective
3. Providing correct parameter values in the right format (integers as numbers, not strings)
4. Evaluating tool results against success/failure criteria
5. Making decisions about next steps based on outcomes

Use the following format:

Question: the workflow step you need to execute
Thought: analyze the objective and determine what needs to be done
Action: choose the most appropriate action from [{tool_names}]
Action Input: provide the necessary input for the action (USE CORRECT DATA TYPES - integers as numbers, not strings with prefixes)
Observation: analyze the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: evaluate if the step objective has been achieved based on success/failure criteria
Final Answer: provide a clear assessment of whether the step succeeded or failed, with reasoning

CRITICAL PARAMETER RULES:
1. For integer parameters, provide ONLY the number (e.g., 935, not "errorcode-935" or "935")
2. For string parameters, provide the actual string value
3. Extract numeric values from context when needed (e.g., if error_code is 935, use 935 for integer parameters)
4. Always match the expected parameter type exactly

CRITICAL RULES:
1. If no suitable tool exists for an objective, respond with "NO_SUITABLE_TOOL"
2. If a tool fails or returns unsatisfactory results, try alternative approaches if available
3. Always evaluate results against the provided success/failure criteria
4. If you cannot achieve the objective after trying available tools, respond with "OBJECTIVE_FAILED"
5. Pay careful attention to parameter types - integers must be numbers, not strings

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        self.react_prompt = PromptTemplate.from_template(react_template)
        
        # Create enhanced ReAct agent
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.available_tools,
            prompt=self.react_prompt
        )
        
        # Create agent executor with better error handling
        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.available_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            early_stopping_method="generate"
        )
        
    def execute_step(self, state: AgentState) -> AgentState:
        """
        Enhanced step execution using ReAct pattern for intelligent decision making.
        """
        if not state["execution_queue"]:
            state["execution_status"] = "completed"
            return state
            
        current_step_id = state["execution_queue"][0]
        state["current_step_id"] = current_step_id
        
        print(f"🔧 Execution Agent: Executing step '{current_step_id}'")
        
        # Handle END step
        if current_step_id.upper() == "END":
            state["execution_status"] = "completed"
            state["execution_queue"] = []
            print("✅ Execution Agent: Workflow completed successfully")
            return state
        
        # Get current step details
        if not state["playbook"] or current_step_id not in state["playbook"]["steps"]:
            if current_step_id.lower() in ['end', 'stop', 'finish', 'complete']:
                state["execution_status"] = "completed"
                state["execution_queue"] = []
                print("✅ Execution Agent: Workflow completed successfully")
                return state
            else:
                state["execution_status"] = "failed"
                state["final_output"]["error"] = f"Step '{current_step_id}' not found in playbook"
                return state
            
        current_step = state["playbook"]["steps"][current_step_id]
        
        try:
            # Execute the step using ReAct pattern
            if current_step["action"] == "achieve_objective":
                result = self._execute_objective_step(current_step, state)
            elif current_step["action"] == "evaluate_condition":
                result = self._evaluate_condition_step(current_step, state)
            elif current_step["action"] == "notify":
                result = self._notify_step(current_step, state)
            else:
                result = {"status": "success", "message": f"Executed step: {current_step['name']}"}
            
            # Check for critical failures
            if result.get("critical_failure"):
                print(f"🛑 Critical failure in step '{current_step_id}': {result.get('message')}")
                state["execution_status"] = "failed"
                state["final_output"]["error"] = result.get("message", "Critical failure occurred")
                return state
            
            # Log the execution
            log_entry = {
                "step_id": current_step_id,
                "step_name": current_step.get("name", ""),
                "objective": current_step.get("objective", ""),
                "result": result,
                "timestamp": "now"
            }
            
            if "execution_log" not in state:
                state["execution_log"] = []
            state["execution_log"].append(log_entry)
            
            # Update context data
            if result.get("context_data"):
                state["context_data"].update(result["context_data"])
            
            # Store last tool result
            state["last_tool_result"] = result
            
            # Determine next step using enhanced logic
            next_step_id = self._determine_next_step(current_step, result)
            
            # Update execution queue
            state["execution_queue"] = state["execution_queue"][1:]
            
            if next_step_id and next_step_id != "END":
                state["execution_queue"].insert(0, next_step_id)
            
            print(f"✅ Step '{current_step_id}' completed. Next: '{next_step_id}'")
            
        except Exception as e:
            print(f"❌ Execution Agent Error in step '{current_step_id}': {str(e)}")
            state["execution_status"] = "failed"
            state["final_output"]["error"] = f"Execution failed at step '{current_step_id}': {str(e)}"
            
        return state
    
    def _execute_objective_step(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """
        Execute an objective-based step using ReAct agent for intelligent tool selection.
        """
        objective = step.get("objective", "")
        success_criteria = step.get("success_criteria", "")
        failure_criteria = step.get("failure_criteria", "")
        context_requirements = step.get("context_requirements", [])
        step_description = step.get("description", "")
        
        # Prepare context information including current error code
        available_context = {
            "error_code": state.get("error_code"),  # Make error code explicitly available
            "current_step": step.get("name", ""),
        }
        
        for req in context_requirements:
            if req in state["context_data"]:
                available_context[req] = state["context_data"][req]
        
        # Create comprehensive question for ReAct agent with parameter guidance
        question = f"""
        WORKFLOW STEP EXECUTION:
        
        Step: {step.get('name', 'Unnamed Step')}
        Objective: {objective}
        Description: {step_description}
        
        SUCCESS CRITERIA: {success_criteria}
        FAILURE CRITERIA: {failure_criteria}
        
        Available Context:
        {json.dumps(available_context, indent=2)}
        
        Previous Step Result:
        {json.dumps(state.get('last_tool_result', {}), indent=2)}
        
        PARAMETER GUIDANCE:
        - If you need to use the error code as an integer parameter, use: {state.get("error_code")}
        - For string parameters, use the actual string value
        - Always match the expected parameter type exactly
        
        Your task is to achieve the objective using the available tools. 
        Evaluate your results against the success/failure criteria.
        If you cannot find suitable tools or achieve the objective, clearly state this in your Final Answer.
        
        REMEMBER: Use correct parameter types - integers as numbers (e.g., {state.get("error_code")}), not strings with prefixes.
        """
        
        try:
            print(f"🤖 ReAct Agent executing objective: {objective}")
            
            # Use ReAct agent to execute the step
            result = self.agent_executor.invoke({"input": question})
            agent_output = result.get('output', '') if hasattr(result, 'get') else str(result)
            
            print(f"🔍 ReAct Agent Output: {agent_output}")
            
            # Enhanced result analysis
            result_analysis = self._analyze_agent_result(agent_output, success_criteria, failure_criteria)
            
            # Check for critical conditions
            if "NO_SUITABLE_TOOL" in agent_output:
                return {
                    "status": "failure",
                    "critical_failure": True,
                    "message": f"No suitable tool available for objective: {objective}",
                    "agent_output": agent_output,
                    "reason": "tool_unavailable"
                }
            
            if "OBJECTIVE_FAILED" in agent_output:
                return {
                    "status": "failure",
                    "critical_failure": True,
                    "message": f"Failed to achieve objective: {objective}",
                    "agent_output": agent_output,
                    "reason": "objective_not_achieved"
                }
            
            # Extract tool results from intermediate steps
            tool_results = []
            context_data = {}
            
            if hasattr(result, 'get') and 'intermediate_steps' in result:
                for step_result in result['intermediate_steps']:
                    if len(step_result) > 1:
                        tool_name = step_result[0].tool if hasattr(step_result[0], 'tool') else 'unknown'
                        tool_output = step_result[1]
                        tool_results.append({
                            "tool": tool_name,
                            "output": tool_output
                        })
                        # Extract potential context data
                        context_data[f"{tool_name}_result"] = tool_output
            
            return {
                "status": result_analysis["status"],
                "objective": objective,
                "success_criteria": success_criteria,
                "failure_criteria": failure_criteria,
                "tool_results": tool_results,
                "agent_output": agent_output,
                "analysis": result_analysis,
                "context_data": context_data,
                "message": result_analysis["message"]
            }
            
        except Exception as e:
            print(f"❌ Error in ReAct execution: {str(e)}")
            return {
                "status": "failure",
                "critical_failure": True,
                "objective": objective,
                "error": str(e),
                "message": f"ReAct execution failed: {str(e)}"
            }
    
    def _analyze_agent_result(self, agent_output: str, success_criteria: str, failure_criteria: str) -> Dict[str, Any]:
        """
        Analyze the ReAct agent output to determine success/failure.
        """
        output_lower = agent_output.lower()
        
        # Enhanced success/failure detection
        success_indicators = ['success', 'completed', 'achieved', 'valid', 'connected', 'verified', 'passed', 'saved']
        failure_indicators = ['failed', 'error', 'invalid', 'disconnected', 'rejected', 'unable', 'cannot']
        
        # Check for explicit success/failure in output
        success_count = sum(1 for indicator in success_indicators if indicator in output_lower)
        failure_count = sum(1 for indicator in failure_indicators if indicator in output_lower)
        
        # Use LLM for more sophisticated analysis if indicators are ambiguous
        if abs(success_count - failure_count) <= 1:  # Ambiguous result
            try:
                analysis_prompt = f"""
                Analyze the following agent output and determine if it indicates success or failure:
                
                Agent Output: {agent_output}
                Success Criteria: {success_criteria}
                Failure Criteria: {failure_criteria}
                
                Respond with exactly: SUCCESS or FAILURE followed by a brief explanation.
                """
                
                analysis_result = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                analysis_text = analysis_result.content.upper()
                
                if "SUCCESS" in analysis_text:
                    return {
                        "status": "success",
                        "message": "Objective achieved based on LLM analysis",
                        "llm_analysis": analysis_result.content
                    }
                else:
                    return {
                        "status": "failure", 
                        "message": "Objective not achieved based on LLM analysis",
                        "llm_analysis": analysis_result.content
                    }
            except:
                pass  # Fall back to simple analysis
        
        # Simple indicator-based analysis
        if success_count > failure_count:
            return {
                "status": "success",
                "message": f"Objective achieved (success indicators: {success_count})",
                "confidence": "indicator_based"
            }
        else:
            return {
                "status": "failure",
                "message": f"Objective not achieved (failure indicators: {failure_count})",
                "confidence": "indicator_based"
            }
    
    def _evaluate_condition_step(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Enhanced condition evaluation using ReAct reasoning."""
        objective = step.get("objective", "")
        last_result = state.get("last_tool_result", {})
        context_data = state.get("context_data", {})
        
        evaluation_question = f"""
        CONDITION EVALUATION TASK:
        
        Objective: {objective}
        
        Previous Step Result: {json.dumps(last_result, indent=2)}
        Available Context: {json.dumps(context_data, indent=2)}
        Error Code: {state.get("error_code")}
        
        Your task is to evaluate whether the condition/objective is met based on the available information.
        Use the available tools if you need to gather additional information.
        
        Provide a clear TRUE or FALSE determination with reasoning.
        """
        
        try:
            result = self.agent_executor.invoke({"input": evaluation_question})
            agent_output = result.get('output', '') if hasattr(result, 'get') else str(result)
            
            # Determine condition result
            condition_met = self._determine_condition_result(agent_output)
            
            return {
                "status": "success" if condition_met else "failure",
                "objective": objective,
                "condition_met": condition_met,
                "agent_output": agent_output,
                "message": f"Condition evaluation: {condition_met}"
            }
            
        except Exception as e:
            return {
                "status": "failure",
                "critical_failure": True,
                "objective": objective,
                "error": str(e),
                "message": f"Condition evaluation failed: {str(e)}"
            }
    
    def _determine_condition_result(self, agent_output: str) -> bool:
        """Determine if condition is met from agent output."""
        output_lower = agent_output.lower()
        
        # Look for explicit true/false statements
        if "true" in output_lower and "false" not in output_lower:
            return True
        if "false" in output_lower and "true" not in output_lower:
            return False
        
        # Look for positive/negative indicators
        positive_indicators = ['yes', 'confirmed', 'met', 'satisfied', 'passed', 'valid']
        negative_indicators = ['no', 'denied', 'not met', 'failed', 'invalid', 'unsatisfied']
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in output_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in output_lower)
        
        return positive_count > negative_count
    
    def _notify_step(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Enhanced notification step."""
        objective = step.get("objective", "")
        description = step.get("description", "")
        
        print(f"📢 Notification: {objective}")
        print(f"   Details: {description}")
        
        return {
            "status": "success",
            "objective": objective,
            "message": f"Notification completed: {objective}"
        }
    
    def _determine_next_step(self, current_step: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Enhanced next step determination."""
        next_steps = current_step.get("next_steps", {})
        result_status = result.get("status", "unknown")
        
        # Check for critical failure - should end workflow
        if result.get("critical_failure"):
            return "END"
        
        # Try to find next step based on result status
        next_step = None
        if result_status in next_steps:
            next_step = next_steps[result_status]
        elif "default" in next_steps:
            next_step = next_steps["default"]
        else:
            next_step = "END"
        
        # Normalize end step variations
        if next_step and next_step.lower() in ['end', 'stop', 'finish', 'complete']:
            next_step = "END"
            
        return next_step


def create_multi_agent_workflow(llm: ChatOpenAI, available_tools: List[BaseTool]) -> StateGraph:
    """
    Creates the enhanced multi-agent workflow with ReAct-powered execution.
    """
    planning_agent = PlanningAgent(llm, available_tools)
    execution_agent = ExecutionAgent(llm, available_tools)
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", planning_agent.analyze_sop)
    workflow.add_node("execution", execution_agent.execute_step)
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Enhanced routing logic
    def route_after_planning(state: AgentState) -> Literal["