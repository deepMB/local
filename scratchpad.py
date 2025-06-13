class ExecutionAgent:
    def execute_step(self, state: AgentState) -> AgentState:  # Keep sync
        # ... existing code until tool execution ...
        
        try:
            # Execute the step using ReAct pattern
            if current_step["action"] == "achieve_objective":
                result = self._execute_objective_step_sync(current_step, state)
            elif current_step["action"] == "evaluate_condition":
                result = self._evaluate_condition_step_sync(current_step, state)
            # ... rest remains same ...
    
    def _execute_objective_step_sync(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Sync wrapper for async execution"""
        try:
            # Use asyncio.run() instead of manual loop management
            result = asyncio.run(self._execute_objective_step_async(step, state))
            return result
        except Exception as e:
            return {
                "status": "failure",
                "critical_failure": True,
                "message": f"Async execution failed: {str(e)}"
            }
    
    def _evaluate_condition_step_sync(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Sync wrapper for async condition evaluation"""
        try:
            result = asyncio.run(self._evaluate_condition_step_async(step, state))
            return result
        except Exception as e:
            return {
                "status": "failure",
                "critical_failure": True,
                "message": f"Async condition evaluation failed: {str(e)}"
            }
    
    async def _execute_objective_step_async(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Actual async implementation"""
        # ... your existing async code with await self.agent_executor.ainvoke() ...
    
    async def _evaluate_condition_step_async(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Actual async implementation"""
        # ... your existing async code with await self.agent_executor.ainvoke() ...

--------------------------

async def main():
    # Initialize MCP client
    client = MultiServerMCPClient({
        "your_server_name": {
            "command": "python",
            "args": ["/path/to/your/mcp_server.py"],
            "transport": "stdio",
        }
    })
    
    # Get tools from MCP servers
    tools = await client.get_tools()
    
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Create workflow
    workflow = create_multi_agent_workflow(llm, tools)
    app = workflow.compile()
    
    # ... initial_state setup ...
    
    try:
        print("🚀 Starting Enhanced Multi-Agent Workflow with ReAct Pattern...")
        print("=" * 70)
        
        # Use regular invoke since workflow nodes are sync
        result = app.invoke(initial_state)  # Back to sync invoke
        
        # ... rest of result processing remains same ...

# Run the async main function
asyncio.run(main())