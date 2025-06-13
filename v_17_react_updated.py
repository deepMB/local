from typing import TypedDict, List, Optional, Any, Dict
from collections import deque
import json
import re
from pydantic import BaseModel, Field

from langgraph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

# Simplified Agent State
class AgentState(TypedDict):
    error_code: int
    sop_content: str
    playbook: Dict[str, Any]
    current_step_id: Optional[str]
    execution_log: List[Dict[str, Any]]
    final_output: Dict[str, Any]
    execution_status: str

class ExecutionAgent:
    """Execution Agent: uses a queue to traverse both IF/ELSE branches and robustly detects tool availability."""
    def __init__(self, llm: ChatOpenAI, tools: List[BaseTool]):
        self.react_agent = create_react_agent(llm, tools)
        self.tools = tools

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        # Initialize queue with the starting step
        step_queue = deque([state.get("current_step_id")])
        visited = set()
        outputs: Dict[str, Any] = {}

        while step_queue:
            step_id = step_queue.popleft()
            if not step_id or step_id.upper() == "END":
                continue
            if step_id in visited:
                continue
            visited.add(step_id)

            step = state["playbook"]["steps"].get(step_id)
            if not step:
                return {"status": "failed", "reason": f"Step '{step_id}' not found."}

            # Build minimal ReAct prompt
            system_prompt = SystemMessage(
                content=(
                    "You are the Execution Agent, a ReAct-style agent. "
                    "Execute one playbook step by calling one tool and decide the next step."
                )
            )
            human_prompt = HumanMessage(
                content=(
                    f"Step ID: {step_id}\n"
                    f"Objective: {step['description']}\n"
                    f"Available tools: {', '.join(t.name for t in self.tools)}\n"
                    f"Context data: {json.dumps(state.get('context_data', {}))}"
                )
            )
            result = self.react_agent.invoke([system_prompt, human_prompt], tools=self.tools)

            # Robust 'no tool' detection
            tool_calls = getattr(result, 'tool_calls', None)
            if not tool_calls:
                return {"status": "terminated", "reason": f"No applicable tool for step {step_id}"}

            # Interpret result
            status = 'success' if result.success else 'failure'
            outputs[step_id] = {
                "status": status,
                "message": result.output,
                "tool_calls": tool_calls
            }

            # Enqueue both branches
            next_map = step.get('next_steps', {})
            for branch in ('success', 'failure'):
                next_id = next_map.get(branch)
                if next_id and next_id.upper() != 'END':
                    step_queue.append(next_id)

            # Abort on critical failure
            if getattr(result, 'critical_failure', False):
                return {"status": "failed", "outputs": outputs}

        return {"status": "completed", "outputs": outputs}

# ... rest of orchestration and PlanningAgent unchanged ...
