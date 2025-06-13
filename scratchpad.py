class ExecutionAgent:
    def execute_step(self, state: AgentState) -> AgentState:  # Keep sync
        # ... existing code until tool execution ...
        
        try:
            # Execute the step using ReAct pattern
            if current_step["action"] == "achieve_objective":
                result = self._execute_objective_step_sync(current_step, state)  # Keep sync wrapper
            elif current_step["action"] == "evaluate_condition":
                result = self._evaluate_condition_step_sync(current_step, state)  # Keep sync wrapper
            # ... rest remains same ...
    
    def _execute_objective_step_sync(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Sync wrapper for async execution"""
        try:
            # Run async code in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._execute_objective_step_async(step, state))
            loop.close()
            return result
        except Exception as e:
            # Handle any async-related errors
            return {
                "status": "failure",
                "critical_failure": True,
                "message": f"Async execution failed: {str(e)}"
            }
    
    async def _execute_objective_step_async(self, step: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
        """Actual async implementation"""
        # ... your existing async code ...
        
        try:
            print(f"🤖 ReAct Agent executing objective: {objective}")
            
            # Use async invoke
            result = await self.agent_executor.ainvoke({"input": question})
            # ... rest of your existing code ...s