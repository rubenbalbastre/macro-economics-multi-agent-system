# Macro Economics Multi-Agent System

## 🧭 Purpose

This repo contains a multi agent system based on LLMs. It incorporates techniques such as RAG and function calling. The agent flow is designed using LangGraph and LangChain.

It is inspired on the [Deep Research](https://github.com/langchain-ai/open_deep_research) and [Deep Agents](https://github.com/langchain-ai/deepagents) projects of LangChain and the [AlphaAgents paper](https://arxiv.org/abs/2508.11152).

The objetive is to be able to analyze the macro-economic context at a given time.


# Status ![Development](https://img.shields.io/badge/⚒️-In_Development-blue)

In progress...


## ⚙️ Installation

The project uses **Python 3.12.3**. Install all dependencies with:

```bash
pip install -r requirements.txt
```

Then, create a `.env` file in the root directory to configure your API keys:

```bash
OPENAI_API_KEY="YOUR_API_KEY"
HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

This enables access to OpenAI models and Hugging Face datasets or uploads.
Once configured, open and run the notebook `app.ipynb`

---

## 🧠 Technical Overview

### examples

Under the directory `/examples/` there are placed some basic scripts to demonstrate the action of each agent individually. Note that this are not tests which intend to cover all the casuistics in an agent behaviour.

## 🛠️ Configuration Example

## Elements

* Monetary Agent
* Labour Market Agent
* Deep Research Agent