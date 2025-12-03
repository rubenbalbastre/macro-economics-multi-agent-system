from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage, filter_messages
from langchain.chat_models import init_chat_model

from typing import Literal, Annotated
from pydantic import Field
import asyncio

from sys import path

path.append("../src/")
from utils import get_buffer_string
from prompts import compress_research_system_prompt, compress_research_human_message, current_state_instructions, future_state_instructions
from tools.others import get_today_str
from subagents.research_lead_agent import ResearchLeadAgent
from tools.think import think_tool
from tools.supervise import ResearchComplete, ConductResearch


# graph state
class MacroAgentState(MessagesState):
    user_query: str
    current_state_messages: Annotated[list[AnyMessage], add_messages] = Field(default=list)
    current_state_summary: str
    future_events_messages: Annotated[list[AnyMessage], add_messages] = Field(default=list)
    future_events_summary: str


# graph flow
class GetCurrentState:

    def __init__(self, llm_config, tools):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model.bind_tools(tools)

    async def __call__(self, state):

        if not state.get("current_state_messages"):
            state.get("current_state_messages").append(HumanMessage(content=state.get("user_query")))

        messages = state.get("current_state_messages")
        
        response = await self.llm.ainvoke(
            input=[
                HumanMessage(content=current_state_instructions.format(
                    messages=get_buffer_string(messages),
                    date=get_today_str()
                ))
            ]
        )

        print("GetCurrentState", response)

        return {"current_state_messages": [response], "messages": [response]}


class GetFutureEvents:

    def __init__(self, llm_config, tools):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model.bind_tools(tools)

    async def __call__(self, state):

        if not state.get("future_events_messages"):
            state["future_events_messages"].extend(
                [
                    HumanMessage(content=future_state_instructions.format(
                        messages="",
                        date=get_today_str(),
                        current_state_summary=state.get("current_state_summary")
                    ))
                ]
            )

        messages = state.get("future_events_messages")
        
        response = await self.llm.ainvoke(
            input=messages
        )
        print("GetFutureEvents", response)
        
        return {"future_events_messages": [response], "messages": [response]}


class Summarizer:

    def __init__(self, llm_config, node_before: Literal["current_state", "future_events"]):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model
        self._node_before = node_before

    async def __call__(self, state: MacroAgentState):
        """Compress research findings into a concise summary.
        
        Takes all the research messages and tool outputs and creates
        a compressed summary suitable for the supervisor's decision-making.
        """
        
        system_message = compress_research_system_prompt.format(date=get_today_str())

        messages_key = "current_state_messages" if self._node_before == "current_state" else "future_events_messages"
        messages = [SystemMessage(content=system_message)] + state.get(messages_key, []) + [HumanMessage(content=compress_research_human_message)]
        response = await self.llm.ainvoke(messages)
               
        summary_key = "current_state_summary" if self._node_before == "current_state" else "future_events_summary"
        return {
            summary_key: str(response.content)
        }


class ToolNode:
    
    def __init__(self, tools, research_tool, macro_step: Literal["current_state", "future_events"]):

        self.research_tool = research_tool
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._macro_step = macro_step

    async def __call__(self, state, config):
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
        messages_key = "current_state_messages" if self._macro_step == "current_state" else "future_events_messages"
        messages = state.get(messages_key, [])
        most_recent_message = messages[-1]
        
        # Initialize variables for single return pattern
        tool_messages = []

        try:

            any_complete_research = [
                tool_call for tool_call in most_recent_message.tool_calls 
                if tool_call["name"] == "ResearchComplete"
            ]

            if any_complete_research:
                tool_results = [{'content': "Research Completed"} for m in any_complete_research]
                tool_messages.extend([
                        ToolMessage(
                            content=result.get("content"),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ) for result, tool_call in zip(tool_results, any_complete_research)
                    ])
            else:
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
                    tool_messages.extend([
                        ToolMessage(
                            content=observation,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    ])

                # Handle ConductResearch calls (asynchronous)
                if conduct_research_calls:
                    # Launch parallel research agents
                    coros = [
                        self.research_tool.ainvoke(input={
                            "messages": [
                                HumanMessage(content=tool_call["args"]["research_topic"])
                            ],
                            "research_topic": tool_call["args"]["research_topic"]
                        },
                        config=config) 
                        for tool_call in conduct_research_calls
                    ]

                    # Wait for all research to complete
                    tool_results = await asyncio.gather(*coros)
                    # tool_results = [{"research_summary": "US economy is fine"} for t in conduct_research_calls]

                    # Format research results as tool messages
                    # Each sub-agent returns compressed research findings in result["compressed_research"]
                    # We write this compressed research as the content of a ToolMessage, which allows
                    # the supervisor to later retrieve these findings via get_notes_from_tool_calls()
                    research_tool_messages = []
                    for result, tool_call in zip(tool_results, conduct_research_calls):
                        result_summary = result.get("research_summary", "Error synthesizing research report")
                        import os
                        if not os.path.isdir(f"data/{tool_call["id"]}"):
                            os.makedirs(f"data/{tool_call["id"]}")
                        with open(f"data/{tool_call["id"]}/summary.md", "w") as f:
                            f.write(result_summary)
                        research_tool_messages.append(
                            ToolMessage(
                            content=result_summary,
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        ))
                    
                    tool_messages.extend(research_tool_messages)

            return {'messages': tool_messages, messages_key: tool_messages}
                    
        except Exception as e:
            print(f"Error in supervisor tools: {e}")


def continue_current_state_search_or_pass_to_future_events_search(state: MacroAgentState):

    last_message = state.get("current_state_messages")[-1]

    if isinstance(last_message, AIMessage):
        path = "current_state_summarizer" if any(
            tool_call["name"] == "ResearchComplete" 
            for tool_call in last_message.tool_calls
            
        ) else "current_state"
    elif isinstance(last_message, ToolMessage):
        path = "current_state_summarizer" if last_message.name == "ResearchComplete" else "current_state"
    return path


def continue_future_events_search_or_end(state: MacroAgentState):

    last_message = state.get("messages")[-1]
    if isinstance(last_message, AIMessage):
        path = "future_events_summarizer" if any(
            tool_call["name"] == "ResearchComplete" 
            for tool_call in last_message.tool_calls
            
        ) else "future_events"
    elif isinstance(last_message, ToolMessage):
        path = "future_events_summarizer" if last_message.name == "ResearchComplete" else "future_events"

    return path


class MacroAgent:

    def __init__(self, llm_config, compile_config):
        self.llm_config = llm_config
        self.compile_config = compile_config
        self.graph = None
        self.compiled_graph = None
        self._research_lead_agent = ResearchLeadAgent(llm_config=self.llm_config, compile_config=self.compile_config)
        self.tools = [
            ConductResearch, ResearchComplete, think_tool
        ]

        self._build_graph()
        self._compile_graph()

    def _build_graph(self):

        graph = StateGraph(MacroAgentState)
        graph.add_node("current_state", GetCurrentState(llm_config=self.llm_config.get("planner").get("current_state"), tools=self.tools))
        graph.add_node("future_events", GetFutureEvents(llm_config=self.llm_config.get("planner").get("future_events"), tools=self.tools))
        graph.add_node("current_state_tools", ToolNode(tools=self.tools, research_tool=self._research_lead_agent, macro_step="current_state"))
        graph.add_node("future_events_tools", ToolNode(tools=self.tools, research_tool=self._research_lead_agent, macro_step="future_events"))
        graph.add_node("current_state_summarizer", Summarizer(llm_config=self.llm_config.get("research").get("summarize_research"), node_before="current_state"))
        graph.add_node("future_events_summarizer", Summarizer(llm_config=self.llm_config.get("research").get("summarize_research"), node_before="future_events"))

        graph.add_edge(START, "current_state")
        graph.add_edge("current_state", "current_state_tools")
        graph.add_conditional_edges("current_state_tools", continue_current_state_search_or_pass_to_future_events_search, {"current_state_summarizer": "current_state_summarizer", "current_state": "current_state"})
        graph.add_edge("current_state_summarizer", "future_events")
        graph.add_edge("future_events", "future_events_tools")
        graph.add_conditional_edges("future_events_tools", continue_future_events_search_or_end, {"future_events_summarizer": "future_events_summarizer", "future_events": "future_events"})
        graph.add_edge("future_events_summarizer", END)

        self.graph = graph
    
    def _compile_graph(self):
        self.compiled_graph = self.graph.compile(**self.compile_config)
    
    async def ainvoke(self, input, config):
        return await self.compiled_graph.ainvoke(input, config=config)
    
    async def __call__(self, input, config):
        return await self.ainvoke(input, config)
    