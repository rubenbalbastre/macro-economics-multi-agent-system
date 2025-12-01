# Macro Economics Multi-Agent System

## 🧭 Purpose

This repo contains a deep research multi agent LLM based system. When asked about a certain topic:

1. It does a first research to save the current state of the topic. 
2. Based on it, it searches for future events which might change it and save them as list.

The motivation behind is that current deep research systems are one-shot tools which process large amounts of information. However, when deploying these systems there is a need to only partial re-run them when certain topics occur. This work proposes a simple 2 stages solution to be used as a base.

This architecture is motivated due to the requirement of knowing past trends and events to be able to answer properly in macro-economic topics. However, this might apply to many different topics.

It is mainly inspired by the [Deep Research](https://github.com/langchain-ai/open_deep_research) and [Deep Agents](https://github.com/langchain-ai/deepagents) projects of LangChain.


## Status ![Development](https://img.shields.io/badge/⚒️-In_Development-blue)

In progress...


## ⚙️ Installation

The project uses **Python 3.12.3**. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Then, create a `.env` file in the root directory to configure your API keys:

```bash
OPENAI_API_KEY="YOUR_API_KEY"
TAVILY_API_KEY="TAVILY_API_KEY"
```

This enables access to OpenAI models and Tavily tools.
Once configured, open and run the notebook `app.ipynb`

---


## 🛠️ Examples

Under the directory `/examples/` there are placed some basic scripts to demonstrate the action of each agent individually. You can find:
* Scope System
* Research Agent
* Lead Research Agent coordinating Research Agents
* Research System (Scope System + Lead Research Agent)


## 🧠 Technical Overview

### System Architecture

#### Full System

The main workflow mirrors the two-stage design described above:

1. **GetCurrentState** → Establish the topic’s present state.
2. **GetFutureState** → Identify future events likely to shift that state.

Both rely on a shared toolkit (“Macro Tools”) that includes:

* **Conduct Research tool** — Launches a Lead Research Agent that performs structured multi-step research.

* **Think tool** — Produces reflection steps that help guide agent reasoning.

* **Complete Research tool** — Signals the end of the research workflow.

![](/docs/images/full_system.png)


#### Research System

Before delegating to the Lead Research Agent, the system first clarifies the research topic. The Lead Research Agent is only triggered once the topic is well-formed and unambiguous.

![](/docs/images/research_system.png)

#### Lead Research Agent

The Lead Research Agent orchestrates one or more Research Agents which is combined using the think tool. Then, it provides a summary.

* **Conduct Research tool**. It launches a Research Agent which uses several tools to reason, search and summarize information of a certain topic.
* **Think tool**. It is used to generate a reflection/reason step of the agent during its execution to guide it in a proper way.
* **Complete Research tool**. It is used to indicate that the research process is finished.

![](/docs/images/lead_research_agent.png)

#### Research Agent

The Research Agent itself uses:

* **Tavily Search tool**. It launches a tavily search run.
* **Think tool**. It is used to generate a reflection/reason step of the agent during its execution to guide it in a proper way.

Then, it provides a summary of what has been found.

![](/docs/images/research_agent.png)


### Statements

This repository contains a set of principales which can be summarized here:

#### Multi-Agent systems in LangGraph

The framework used to develop this project was LangGraph due to the flexibility it offers and my familiarity with it. However, there is no specific reason that might block anyone to implement it using another framework.

#### Workflows/Agents as Objects
 
To facilitate configuration of different agents and tools, a yaml config file is provided. To work on top of that, a object oriented style is followed to implement Workflows and Agent. Classes are created to hold logics and instances are specific LLM configurations of those classes.

LLMs configuration is defined in a yaml file similar to this. It contains the model name and the temperature. Note that this could extend to other parameters but it was not required to the purpose of this repo.

```yaml
"research":
  "research_agent":
    "model_name": "gpt-4o-mini"
    "temperature": 0
  "summarize_research":
    "model_name": "gpt-4o-mini"
    "temperature": 0
"scope":
  "topic_clarification":
    "model_name": "gpt-4o-mini"
    "temperature": 0
  "research_brief":
    "model_name": "gpt-4o-mini"
    "temperature": 0
"supervisor":
  "supervisor_agent":
    "model_name": "gpt-4o-mini"
    "temperature": 0
"planner":
  "current_state":
    "model_name": "gpt-4o-mini"
    "temperature": 0
  "future_events":
    "model_name": "gpt-4o-mini"
    "temperature": 0
```

#### Tavily Search as Search Engine

The search engine used is tavily search. This is motivated by two reasons:

1. The project does not aim to be very complex to execute but just be a proof of concept. Then, a good idea is to try to reduce the number of items and reuse existing tools.

2. Tavily services reduce token usage since they serve LLM summaries of webpages.

3. Rely on tavily search engine results. No tavily extract engine nor custom web scrapping is done. We aim to get information which is accesible through quick summaries. This might not be the case always but here we assume it.

## Future Potential Developments

* A system which triggers automatically a new research run when a future events occurs.


## References:


* [Deep Research](https://github.com/langchain-ai/open_deep_research)
* [Deep Agents](https://github.com/langchain-ai/deepagents)
* [AlphaAgents paper](https://arxiv.org/abs/2508.11152)
* [Antrophic "think" tool](https://www.anthropic.com/engineering/claude-think-tool)
* [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)