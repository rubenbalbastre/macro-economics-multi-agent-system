from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, filter_messages, ToolMessage
from langchain.chat_models import init_chat_model

import operator
from typing import Annotated, Literal, List
from pydantic import BaseModel, Field

from sys import path

path.append("../src/")
from prompts import compress_research_system_prompt, research_agent_prompt, compress_research_human_message
from tools.others import get_today_str
from tools.search import tavily_search
from tools.think import think_tool


# graph state
class ResearchAgentState(MessagesState):
    number_of_tool_calls: int
    research_topic: str
    research_summary: str
    raw_notes: Annotated[List[str], operator.add]


# structured output schemas
class ResearchSummary(BaseModel):
    summary: str = Field(description="Concise sumary of the webpage content")


class LLMCall:
    def __init__(self, llm_config, tools):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm_with_tools = model.bind_tools(tools)

    async def __call__(self, state: ResearchAgentState):
        """Analyze current state and decide on next actions.
        
        The model analyzes the current conversation state and decides whether to:
        1. Call search tools to gather more information
        2. Provide a final answer based on gathered information
        
        Returns updated state with the model's response.
        """
        return {
            "messages": [
                await self.llm_with_tools.ainvoke(
                    [SystemMessage(content=research_agent_prompt)] + state["messages"]
                )
            ]
        }


class SummarizeResearch:
    def __init__(self, llm_config):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model

    async def __call__(self, state: ResearchAgentState):
        """Compress research findings into a concise summary.
        
        Takes all the research messages and tool outputs and creates
        a compressed summary suitable for the supervisor's decision-making.
        """
        
        system_message = compress_research_system_prompt.format(date=get_today_str())
        messages = [SystemMessage(content=system_message)] + state.get("messages", []) + [HumanMessage(content=compress_research_human_message)]
        response = await self.llm.ainvoke(messages)
        
        # Extract raw notes from tool and AI messages
        raw_notes = [
            str(m.content) for m in filter_messages(
                state["messages"], 
                include_types=["tool", "ai"]
            )
        ]
        
        return {
            "research_summary": str(response.content),
            "raw_notes": ["\n".join(raw_notes)]
        }


class ToolNode:
    def __init__(self, tools):
        # Set up tools
        self.tools_by_name = {tool.name: tool for tool in tools}


    async def __call__(self, state: ResearchAgentState):
        """Execute all tool calls from the previous LLM response.
        
        Executes all tool calls from the previous LLM responses.
        Returns updated state with tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
    
        # Execute all tool calls
        observations = []
        for tool_call in tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            observations.append(
                await tool.ainvoke(tool_call["args"])
            )
                
        # Create tool message outputs
        tool_outputs = [
            ToolMessage(
                content=observation,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ) for observation, tool_call in zip(observations, tool_calls)
        ]
        
        return {"messages": tool_outputs}


def route_research(state: ResearchAgentState) -> Literal["tool_node", "summarize_research"]:

    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM makes a tool call, continue to tool execution
    if last_message.tool_calls:
        path = "tool_node"
    elif len(messages) > 6:
        path = "summarize_research"
    else:
        path = "summarize_research"
    
    return path


class ResearchAgent:
    def __init__(self, llm_config, compile_config = {}):
        self.llm_config = llm_config
        self.graph = None
        self.compiled_graph = None

        self._build_graph()
        self._compile_graph(compile_config)

    def _build_graph(self):

        tools = [think_tool, tavily_search]

        graph = StateGraph(ResearchAgentState)

        graph.add_node("llm_call", LLMCall(llm_config=self.llm_config.get("research_agent"), tools=tools))
        graph.add_node("tool_node", ToolNode(tools=tools))
        graph.add_node("summarize_research", SummarizeResearch(llm_config=self.llm_config.get("summarize_research")))

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", route_research, {"tool_node": "tool_node", "summarize_research": "summarize_research"})
        graph.add_edge("tool_node", "llm_call")
        graph.add_edge("summarize_research", END)

        self.graph = graph

    def _compile_graph(self, compile_config):
        self.compiled_graph = self.graph.compile(**compile_config)
    
    async def ainvoke(self, input, config = {}):
        return await self.compiled_graph.ainvoke(input, config=config)
    
    async def __call__(self, input, config):
        return await self.ainvoke(input, config)
    