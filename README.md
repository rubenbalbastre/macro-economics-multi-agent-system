# Macro Economics Multi-Agent System

## 🧭 Purpose

A macro economic multi agent LLM based system. It contains a deep research system combined with a persistent memory tracking system. When asked about a certain topic, it initiates a topic to track. It does a first research to save its current state. Then, when triggered (it could be done periodically) it runs another research to update its past state.

This architecture is motivated due to the requirement of knowing past trends and events to be able to answer properly in macro-economic topics. However, this might apply to many different topics.

It is mainly inspired by the [Deep Research](https://github.com/langchain-ai/open_deep_research) and [Deep Agents](https://github.com/langchain-ai/deepagents) projects of LangChain and the [AlphaAgents paper](https://arxiv.org/abs/2508.11152).


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
HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

This enables access to OpenAI models and Hugging Face datasets or uploads.
Once configured, open and run the notebook `app.ipynb`

---


## 🛠️ Examples

Under the directory `/examples/` there are placed some basic scripts to demonstrate the action of each agent individually. You can find:

* Research System Examples:
  * Scope System
  * Research Agent
  * Research Supervisor Agent + Research Agents
* 


## 🧠 Technical Overview

### Introduction

This repository contains a set of principales which can be summarized here:

![](/docs/images/multi-agent-system.png)


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
```

### System Components

#### Deep Research Agent

A system to perform web searches in a loop to be able to find information about a certain topic.

#### Tracking Agent



#### Monetary Agent
#### Labour Market Agent


## References:


* [Deep Research](https://github.com/langchain-ai/open_deep_research)
* [Deep Agents](https://github.com/langchain-ai/deepagents)
* [AlphaAgents paper](https://arxiv.org/abs/2508.11152)
* [Antrophic "think" tool](https://www.anthropic.com/engineering/claude-think-tool)
* [Anthropic Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system)