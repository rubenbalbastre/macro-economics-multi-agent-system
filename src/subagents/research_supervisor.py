import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from sys import path
path.append("../src/")

from prompts import lead_researcher_prompt
from tools.supervise import (
    ConductResearch, 
    ResearchComplete
)
from tools.others import get_today_str
from tools.think import think_tool
import operator
from typing_extensions import Annotated

from langchain_core.messages import BaseMessage


class SupervisorState(MessagesState):
    """
    State for the multi-agent research supervisor.
    
    Manages coordination between supervisor and research agents, tracking
    research progress and accumulating findings from multiple sub-agents.
    """
    
    # Detailed research brief that guides the overall research direction
    research_brief: str
    # Processed and structured notes ready for final report generation
    notes: Annotated[list[str], operator.add] = []
    # Counter tracking the number of research iterations performed
    research_iterations: int = 0
    # Raw unprocessed research notes collected from sub-agent research
    raw_notes: Annotated[list[str], operator.add] = []


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """Extract research notes from ToolMessage objects in supervisor message history.
    
    This function retrieves the compressed research findings that sub-agents
    return as ToolMessage content. When the supervisor delegates research to
    sub-agents via ConductResearch tool calls, each sub-agent returns its
    compressed findings as the content of a ToolMessage. This function
    extracts all such ToolMessage content to compile the final research notes.
    
    Args:
        messages: List of messages from supervisor's conversation history
        
    Returns:
        List of research note strings extracted from ToolMessage objects
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]

# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    # Only apply if running in Jupyter/IPython environment
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not in Jupyter, no need for nest_asyncio
except ImportError:
    pass  # nest_asyncio not available, proceed without it

# ===== SUPERVISOR NODES =====

class LLMCall:
    def __init__(self, llm_config, tools):
        llm = init_chat_model(model="gpt-4o-mini", temperature=0)
        self.llm_with_tools = llm.bind_tools(tools)
        # Maximum number of concurrent research agents the supervisor can launch
        # This is passed to the lead_researcher_prompt to limit parallel research tasks
        self.max_concurrent_researchers = 3
        # Maximum number of tool call iterations for individual researcher agents
        # This prevents infinite loops and controls research depth per topic
        self.max_researcher_iterations = 6 # Calls to think_tool + ConductResearch

    async def __call__(self, state: SupervisorState) -> Command[Literal["tool_node"]]:
        """Coordinate research activities.
        
        Analyzes the research brief and current progress to decide:
        - What research topics need investigation
        - Whether to conduct parallel research
        - When research is complete
        
        Args:
            state: Current supervisor state with messages and research progress
            
        Returns:
            Command to proceed to tool_node node with updated state
        """
        messages = state.get("messages", [])
        
        # Prepare system message with current date and constraints
        system_message = lead_researcher_prompt.format(
            date=get_today_str(), 
            max_concurrent_research_units=self.max_concurrent_researchers,
            max_researcher_iterations=self.max_researcher_iterations
        )
        messages = [SystemMessage(content=system_message)] + messages
        
        # Make decision about next research steps
        response = await self.llm_with_tools.ainvoke(messages)
        
        return Command(
            goto="tool_node",
            update={
                "messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1
            }
        )


class ToolNode:
    def __init__(self, tools, research_tool):

        self.research_tool = research_tool
        # Set up tools
        self.tools_by_name = {tool.name: tool for tool in tools}

        # Maximum number of tool call iterations for individual researcher agents
        # This prevents infinite loops and controls research depth per topic
        self.max_researcher_iterations = 6 # Calls to think_tool + ConductResearch

    async def __call__(self, state: SupervisorState) -> Command[Literal["supervisor", END]]:
        """Execute supervisor decisions - either conduct research or end the process.
        
        Handles:
        - Executing think_tool calls for strategic reflection
        - Launching parallel research agents for different topics
        - Aggregating research results
        - Determining when research is complete
        
        Args:
            state: Current supervisor state with messages and iteration count
            
        Returns:
            Command to continue supervision, end process, or handle errors
        """
        messages = state.get("messages", [])
        research_iterations = state.get("research_iterations", 0)
        most_recent_message = messages[-1]
        
        # Initialize variables for single return pattern
        tool_messages = []
        all_raw_notes = []
        next_step = "supervisor"  # Default next step
        should_end = False
        
        # Check exit criteria first
        exceeded_iterations = research_iterations >= self.max_researcher_iterations
        no_tool_calls = not most_recent_message.tool_calls
        research_complete = any(
            tool_call["name"] == "ResearchComplete" 
            for tool_call in most_recent_message.tool_calls
        )
        
        if exceeded_iterations or no_tool_calls or research_complete:
            should_end = True
            next_step = END
        
        else:
            # Execute ALL tool calls before deciding next step
            try:
                # Separate think_tool calls from ConductResearch calls
                think_tool_calls = [
                    tool_call for tool_call in most_recent_message.tool_calls 
                    if tool_call["name"] == "think_tool"
                ]
                
                conduct_research_calls = [
                    tool_call for tool_call in most_recent_message.tool_calls 
                    if tool_call["name"] == "ConductResearch"
                ]

                # Handle think_tool calls (synchronous)
                for tool_call in think_tool_calls:
                    observation = self.tools_by_name["think_tool"].invoke(tool_call["args"])
                    tool_messages.append(
                        ToolMessage(
                            content=observation,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )

                # Handle ConductResearch calls (asynchronous)
                if conduct_research_calls:
                    # Launch parallel research agents
                    coros = [
                        self.research_tool.ainvoke({
                            "messages": [
                                HumanMessage(content=tool_call["args"]["research_topic"])
                            ],
                            "research_topic": tool_call["args"]["research_topic"]
                        }) 
                        for tool_call in conduct_research_calls
                    ]

                    # Wait for all research to complete
                    tool_results = await asyncio.gather(*coros)

                    # Format research results as tool messages
                    # Each sub-agent returns compressed research findings in result["compressed_research"]
                    # We write this compressed research as the content of a ToolMessage, which allows
                    # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
                    research_tool_messages = [
                        ToolMessage(
                            content=result.get("research_summary", "Error synthesizing research report"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for result, tool_call in zip(tool_results, conduct_research_calls)
                    ]
                    
                    tool_messages.extend(research_tool_messages)

                    # Aggregate raw notes from all research
                    all_raw_notes = [
                        "\n".join(result.get("raw_notes", [])) 
                        for result in tool_results
                    ]
                    
            except Exception as e:
                print(f"Error in supervisor tools: {e}")
                should_end = True
                next_step = END
        
        # Single return point with appropriate state updates
        if should_end:
            return Command(
                goto=next_step,
                update={
                    "notes": get_notes_from_tool_calls(messages),
                    "research_brief": state.get("research_brief", "")
                }
            )
        else:
            return Command(
                goto=next_step,
                update={
                    "messages": tool_messages,
                    "raw_notes": all_raw_notes
                }
            )

class Supervisor:
    def __init__(self, llm_config, compile_config):

        self.llm_config = llm_config
        self.graph = None
        self.compiled_graph = None

        self._build_graph()
        self._compile_graph(compile_config)

    def _build_graph(self):

        tools = [ConductResearch, ResearchComplete, think_tool]
        
        from subagents.research_agent import Research
        research_tool = Research(
            llm_config=self.llm_config.get("research")
        )

        graph = StateGraph(SupervisorState)

        graph.add_node("supervisor", LLMCall(llm_config=self.llm_config.get("supervisor").get("supervisor_agent"), tools=tools))
        graph.add_node("tool_node", ToolNode(tools=tools, research_tool=research_tool))

        graph.add_edge(START, "supervisor")

        self.graph = graph

    def _compile_graph(self, compile_config):
        self.compiled_graph = self.graph.compile(**compile_config)
    
    async def ainvoke(self, input, config):
        return await self.compiled_graph.ainvoke(input, config=config)
