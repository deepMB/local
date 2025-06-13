class ExecutionAgent:
    """
    Enhanced Execution Agent that uses ReAct pattern for intelligent tool selection and decision making.
    """
    
    def __init__(self, llm: ChatOpenAI, available_tools: List[BaseTool]):
        self.llm = llm
        self.available_tools = available_tools
        self.tool_dict = {tool.name: tool for tool in available_tools}
        
        # Get tool names for the prompt
        tool_names = [tool.name for tool in available_tools]
        
        # Enhanced ReAct prompt for workflow execution with better argument formatting
        react_template = """You are an intelligent workflow execution agent. You have access to the following tools:

{tools}

Your role is to execute workflow steps by:
1. Understanding the objective of each step
2. Choosing the most appropriate tool(s) to achieve the objective
3. Evaluating tool results against success/failure criteria
4. Making decisions about next steps based on outcomes

Use the following format:

Question: the workflow step you need to execute
Thought: analyze the objective and determine what needs to be done
Action: choose the most appropriate action from [{tool_names}]
Action Input: the input to the action
Observation: analyze the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: evaluate if the step objective has been achieved based on success/failure criteria
Final Answer: provide a clear assessment of whether the step succeeded or failed, with reasoning

CRITICAL RULES FOR ACTION INPUT:
1. For tools with single parameters, provide just the value (e.g., for errorcode parameter, use: 935)
2. For tools with multiple parameters, provide a valid JSON object (e.g., {{"errorcode": 935, "required_fields": ["field1", "field2"]}})
3. Ensure integer parameters are passed as numbers, not strings
4. Ensure string parameters are properly quoted when in JSON
5. Ensure list parameters are properly formatted as JSON arrays when in JSON
6. Match the exact parameter names expected by the tool

ADDITIONAL RULES:
1. If no suitable tool exists for an objective, respond with "NO_SUITABLE_TOOL"
2. If a tool fails or returns unsatisfactory results, try alternative approaches if available
3. Always evaluate results against the provided success/failure criteria
4. If you cannot achieve the objective after trying available tools, respond with "OBJECTIVE_FAILED"

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
            max_iterations=5,  # Reduced to prevent excessive iterations
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
        
        # Prepare context information
        available_context = {}
        for req in context_requirements:
            if req in state["context_data"]:
                available_context[req] = state["context_data"][req]
        
        # Get error code from state for tool calls that need it
        error_code = state.get("error_code", 0)
        
        # Create comprehensive question for ReAct agent
        question = f"""
        WORKFLOW STEP EXECUTION:
        
        Step: {step.get('name', 'Unnamed Step')}
        Objective: {objective}
        Description: {step_description}
        
        SUCCESS CRITERIA: {success_criteria}
        FAILURE CRITERIA: {failure_criteria}
        
        ERROR CODE CONTEXT: {error_code} (Use this value for any errorcode parameters)
        
        Available Context:
        {json.dumps(available_context, indent=2) if available_context else "No specific context required"}
        
        Previous Step Result:
        {json.dumps(state.get('last_tool_result', {}), indent=2)}
        
        IMPORTANT PARAMETER FORMATTING:
        - For errorcode parameters: Use the integer {error_code}
        - For single parameter tools: Provide just the value
        - For multi-parameter tools: Use JSON format like {{"param1": value1, "param2": value2}}
        
        Your task is to achieve the objective using the available tools. 
        Evaluate your results against the success/failure criteria.
        If you cannot find suitable tools or achieve the objective, clearly state this in your Final Answer.
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
        success_indicators = ['success', 'completed', 'achieved', 'valid', 'connected', 'verified', 'passed']
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