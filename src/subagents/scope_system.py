from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, get_buffer_string, AIMessage
from langchain.chat_models import init_chat_model

from pydantic import BaseModel, Field
import json

from sys import path

path.append("../src/")
from prompts import clarify_with_user_instructions, transform_messages_into_research_topic_prompt
from tools.others import get_today_str

# graph state
class ScopeSystemState(MessagesState):
    is_topic_clarified: bool


# structured output schemas
class ResearchQuestion(BaseModel):
    title: str


class ResearchTopicAnalysis(BaseModel):
    is_topic_clarified: bool = Field(
        description="Whether the user needs to be asked for a clarification on the topic."
    )
    question: str = Field(
        description="A question to ask the user to clarify the topic research."
    )


class TopicClarifier:

    def __init__(self, llm_config):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model.with_structured_output(ResearchTopicAnalysis)

    async def __call__(self, state):
        response = await self.llm.ainvoke(
            input=[
                HumanMessage(content=clarify_with_user_instructions.format(
                    messages=get_buffer_string(state.get("messages")),
                    date=get_today_str()
                ))
            ]
        )
        ai_message = AIMessage(content=json.dumps(response.model_dump()))
        is_topic_clarified = response.is_topic_clarified

        return {"messages": [ai_message], "is_topic_clarified": is_topic_clarified}


class ResearchBrief:

    def __init__(self, llm_config):
        self._llm_config = llm_config
        model = init_chat_model(
            model=self._llm_config.get("model_name"), 
            temperature=self._llm_config.get("temperature")
        )
        self.llm = model.with_structured_output(ResearchQuestion)

    async def __call__(self, state):
        response = await self.llm.ainvoke([
        HumanMessage(
            content=transform_messages_into_research_topic_prompt.format(
                messages=get_buffer_string(state.get("messages", [])),
                date=get_today_str()
            ))
        ])

        ai_message = AIMessage(content=json.dumps(response.model_dump()))

        return {"messages": [ai_message]}


def check_clarity(state):
    direction = "write_research_brief" if state.get("is_topic_clarified") else END
    return direction


class ScopeSystem:
    def __init__(self, llm_config, compile_config):
        self.llm_config = llm_config
        self.graph = None
        self.compiled_graph = None

        self._build_graph()
        self._compile_graph(compile_config)

    def _build_graph(self):

        graph = StateGraph(ScopeSystemState)
        graph.add_node("analyze_research_topic", TopicClarifier(llm_config=self.llm_config.get("topic_clarification")))
        graph.add_node("write_research_brief", ResearchBrief(llm_config=self.llm_config.get("research_brief")))

        graph.add_edge(START, "analyze_research_topic")
        graph.add_conditional_edges("analyze_research_topic", check_clarity, {"write_research_brief": "write_research_brief", END: END})
        graph.add_edge("write_research_brief", END)

        self.graph = graph
    
    def _compile_graph(self, compile_config):
        self.compiled_graph = self.graph.compile(**compile_config)
    
    async def ainvoke(self, input, config):
        return await self.compiled_graph.ainvoke(input, config=config)
