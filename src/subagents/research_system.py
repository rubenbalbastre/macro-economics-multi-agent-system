from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Optional, Sequence, Annotated
from langchain_core.messages import BaseMessage
import operator

from subagents.scope_system import TopicClarifier, ResearchBrief, check_clarity
from subagents.research_lead_agent import ResearchLeadAgent


class ResearchSystemState(MessagesState):

    # initial user query
    user_query: str
    # Research brief generated from user conversation history
    research_brief: Optional[str]
    # Messages exchanged with the supervisor agent for coordination
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Processed and structured notes ready for report generation
    notes: Annotated[list[str], operator.add] = []
    # Final formatted research report
    final_report: str


class ResearchSystem:

    def __init__(self, llm_config, compile_config):
        self.llm_config = llm_config
        self.compile_config = compile_config
        self.graph = None
        self.compiled_graph = None

        self._build_graph()
        self._compile_graph()

    def _build_graph(self):

        graph = StateGraph(ResearchSystemState)
        graph.add_node("topic_clarification", TopicClarifier(llm_config=self.llm_config.get("scope").get("topic_clarification")))
        graph.add_node("write_research_brief", ResearchBrief(llm_config=self.llm_config.get("scope").get("research_brief")))
        graph.add_node("lead_research_agent", ResearchLeadAgent(llm_config=self.llm_config, compile_config=self.compile_config))

        graph.add_edge(START, "topic_clarification")
        graph.add_conditional_edges("topic_clarification", check_clarity, {"write_research_brief": "write_research_brief", END: END})
        graph.add_edge("write_research_brief", "lead_research_agent")
        graph.add_edge("lead_research_agent", END)

        self.graph = graph
    
    def _compile_graph(self):
        self.compiled_graph = self.graph.compile(**self.compile_config)
    
    async def ainvoke(self, input, config):
        return await self.compiled_graph.ainvoke(input, config=config)

    async def __call__(self, input, config):
        return await self.ainvoke(input, config)
    