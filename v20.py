# Lesson 6: Multi-Agent System with LangGraph - ReAct Pattern

from dotenv import load_dotenv
import json
import re
from typing import Dict, Any, List, TypedDict, Annotated
import operator

# Load environment variables
_ = load_dotenv()

# Import required LangGraph and LangChain components
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Initialize memory
memory = SqliteSaver.from_conn_string(":memory:")

# Define the agent state - simplified for ReAct pattern
class AgentState(TypedDict):
    task: str
    plan: Dict[str, Any]
    messages: Annotated[List[AnyMessage], operator.add]  # This will accumulate all messages including tool outputs
    error_code: int
    sop_content: str
    available_tools: List[str]

# Initialize the LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define some example tools
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock implementation - replace with actual web search
    return f"Mock search results for: {query}. Found relevant information about the topic."

@tool
def document_analyzer(content: str) -> str:
    """Analyze document content."""
    # Mock implementation - replace with actual document analysis
    return f"Analysis of document content: The content appears to be about {content[:50]}... Key insights extracted."

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)  # Note: In production, use safer evaluation
        return str(result)
    except Exception as e:
        return f"Error in calculation: {str(e)}"

# Create tools dictionary
tools = [web_search, document_analyzer, calculator]
tools_dict = {tool.name: tool for tool in tools}

def _extract_json_from_response(response_content: str) -> Dict[str, Any]:
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

def planner(state: AgentState) -> AgentState:
    """Creates a plan based on the task and initializes the conversation."""
    task = state["task"]
    
    # Create tool descriptions
    tool_descriptions = []
    for tool in tools:
        tool_descriptions.append(f"- {tool.name}: {tool.description}")
    
    planner_prompt = f"""
    You are an expert task planner that creates step-by-step execution plans.
    
    AVAILABLE TOOLS:
    {chr(10).join(tool_descriptions)}
    
    Create a JSON plan for the given task. Each step should define clear objectives.
    
    IMPORTANT: Return ONLY the JSON plan, no other text or explanations.
    """
    
    human_prompt = f"""
    Create a step-by-step JSON plan for this task: {task}
    
    Required JSON structure:
    {{
        "name": "Plan name",
        "description": "What this plan accomplishes", 
        "steps": [
            {{
                "id": "step_1",
                "name": "Step name",
                "objective": "What this step needs to accomplish",
                "success_criteria": "How to determine success"
            }}
        ]
    }}
    
    Create a practical plan with 2-3 steps maximum.
    """
    
    messages = [
        SystemMessage(content=planner_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    response = model.invoke(messages)
    
    try:
        plan_json = _extract_json_from_response(response.content)
        state["plan"] = plan_json
        
        # Initialize the conversation with the plan
        initial_message = HumanMessage(content=f"""
        Task: {task}
        
        Plan created: {plan_json['name']}
        Description: {plan_json['description']}
        
        Steps to execute:
        {chr(10).join([f"{i+1}. {step['name']}: {step['objective']}" for i, step in enumerate(plan_json.get('steps', []))])}
        
        Please execute this plan step by step. Use the available tools when needed.
        """)
        
        return {"messages": [initial_message]}
        
    except Exception as e:
        print(f"Error parsing plan: {e}")
        # Create a simple fallback plan
        fallback_message = HumanMessage(content=f"""
        Task: {task}
        
        Please complete this task using the available tools: {[tool.name for tool in tools]}
        """)
        
        return {"messages": [fallback_message]}

def should_continue(state: AgentState) -> bool:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    return hasattr(last_message, 'tool_calls') and last_message.tool_calls and len(last_message.tool_calls) > 0

def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state - ReAct pattern."""
    llm_with_tools = model.bind_tools(tools)
    
    react_system_prompt = f"""You are an intelligent ReAct agent that can use tools to accomplish tasks.

Available tools: {[tool.name for tool in tools]}

Use the ReAct format:
- Thought: Think about what you need to do
- Action: Choose a tool to use (or decide you're done)
- Action Input: Provide input for the tool
- Observation: You'll receive the tool result
- ... (repeat Thought/Action/Action Input/Observation as needed)
- Final Answer: Provide your final response when the task is complete

IMPORTANT:
- Use tools when you need external information or calculations
- If you can answer without tools, provide a direct response
- When the task is complete, give a "Final Answer" without making tool calls
- Be concise but thorough
"""

    # Prepare messages with system prompt + conversation history
    messages = [SystemMessage(content=react_system_prompt)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response and add results to messages."""
    last_message = state['messages'][-1]
    
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return {"messages": []}
    
    tool_calls = last_message.tool_calls
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call.get('args', {})
        
        print(f"Calling Tool: {tool_name} with args: {tool_args}")
        
        if tool_name not in tools_dict:
            print(f"Tool: {tool_name} does not exist.")
            result = f"Error: Tool '{tool_name}' does not exist. Available tools: {list(tools_dict.keys())}"
        else:
            try:
                # Get the first argument value (assuming single argument tools for simplicity)
                arg_value = list(tool_args.values())[0] if tool_args else ""
                result = tools_dict[tool_name].invoke(arg_value)
                print(f"Tool result: {result}")
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
        
        # Create tool message that will be added to the conversation
        tool_message = ToolMessage(
            tool_call_id=tool_call['id'], 
            name=tool_name, 
            content=str(result)
        )
        tool_messages.append(tool_message)
    
    print("Tools execution complete. Results added to conversation.")
    return {"messages": tool_messages}

# Build the graph - Simple ReAct pattern
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("planner", planner)
graph.add_node("llm", call_llm)
graph.add_node("tools", take_action)

# Add edges - ReAct pattern: planner -> llm -> (tools or END)
graph.add_edge("planner", "llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "tools", False: END}
)
graph.add_edge("tools", "llm")  # After tools, go back to LLM

# Set entry point
graph.set_entry_point("planner")

# Compile the graph
agent = graph.compile(checkpointer=memory)

# Test the agent
if __name__ == "__main__":
    print("Multi-Agent ReAct System initialized successfully!")
    
    # Example usage
    thread = {"configurable": {"thread_id": "1"}}
    
    # Test with a simple task
    initial_state = {
        'task': "Calculate the square root of 144 and then search for information about that number",
        'messages': [],
        'plan': {},
        'error_code': 0,
        'sop_content': "",
        'available_tools': [tool.name for tool in tools]
    }
    
    print("\n" + "="*50)
    print("STARTING REACT AGENT EXECUTION")
    print("="*50)
    
    for step_output in agent.stream(initial_state, thread):
        node_name = list(step_output.keys())[0]
        node_output = step_output[node_name]
        
        print(f"\n--- {node_name.upper()} NODE ---")
        
        if 'messages' in node_output and node_output['messages']:
            for msg in node_output['messages']:
                if hasattr(msg, 'content'):
                    print(f"Message: {msg.content[:200]}...")
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
        
    print("\n" + "="*50)
    print("REACT AGENT EXECUTION COMPLETED")
    print("="*50)