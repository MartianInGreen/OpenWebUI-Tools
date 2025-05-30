# OpenWebUI-Tools

OpenWebUI-Tools is a comprehensive toolkit designed to enhance the capabilities of OpenWebUI. This collection includes powerful tools for web scraping, knowledge querying, web searching, and YouTube video analysis, as well as an advanced Sequential Multi-Agent Reasoning Technique (SMART) system.

The SMART system elevates the project by providing a sophisticated multi-agent approach to handling complex queries and tasks. It combines planning, reasoning, tool use, and user interaction in a sequential pipeline, allowing for more nuanced and effective responses to user inputs.

This toolkit aims to provide developers with a versatile set of resources to build more intelligent and capable web-based AI applications.

## ReAct 

---
**Install instructions:**
- Simply copy the `smart.py` code into a new function in OpenWebUI.
- To create a new fuction go to `Workspaces -> Functions -> New Function (+ Icon)`

NOTE: Some implemented tools need non avalible external APIs. 

---

The ReAct system is a pipeline for automatic model and tool selection as well as model tool calling. 

## SMART - Sequential Multi-Agent Reasoning Technique

---
**Install instructions:**
- Simply copy the `smart.py` code into a new function in OpenWebUI.
- To create a new fuction go to `Workspaces -> Functions -> New Function (+ Icon)`

NOTE: I would not recommend using the smart_with_tools.py code _as is_. It implements my own tools, many of which don't work without some external self-hosted APIs. However you can use it as a template to automatically implement your own tools.

---

The SMART (Sequential Multi-Agent Reasoning Technique) system is a powerful pipeline for enhanced language model capabilities. It includes:

- Planning Agent: Prepares incoming user requests for subsequent agents in the chain.
- Reasoning Agent: Handles internal thought processes, planning, and thinking.
- Tool-use Agent: Interfaces with various tools to gather information or perform actions.
- User-interaction Agent: Manages direct communication with the user.

Key features:
- Adaptive model selection based on task complexity
- Multi-step reasoning process
- Integration with external tools
- Customizable prompts for each agent in the chain
- Advanced Planning and internal thought 

SMART enhances the problem-solving capabilities of language models by breaking down complex tasks into manageable steps and leveraging specialized agents for different aspects of the process.

## Available Tools

### 1. Scrape Website (scrape.py)
- Scrapes any website and returns readable, Markdown-formatted results.
- Optimized for LLM processing.

### 2. WolframAlpha API (wolfram.py)
- Queries the WolframAlpha knowledge engine for a wide variety of questions.
- Supports real-time data queries, mathematical equations, scientific questions, and more.

### 3. Web Search (webSearch.py)
- Performs web searches using the Brave Search API.
- Supports various search focuses including web, news, Wikipedia, academia, Reddit, images, and videos.

### 4. YouTube Tool (youtube.py)
- Retrieves information about YouTube videos, including metadata and transcriptions. Supports multiple languages for video transcripts.
- Searches YouTube by keyword (1–50 results), returning for each video its id and metadata.
- Uses YouTube API v3 and needs a free api key from Google Cloud. https://developers.google.com/youtube/v3/getting-started

**Installation**

To install these tools, simply copy the code into a new tool inside [OpenWeb UI](https://www.openwebui.com).

**Usage**

Each tool can be imported and used in your OpenWebUI projects. Refer to the individual tool files for specific usage instructions and required API keys.

## Project Info

**Contributing**

Contributions to OpenWebUI-Tools are welcome! Please feel free to submit pull requests or open issues for any improvements or bug fixes.

**License**

Licensed under the MIT License. 

**Authors**

[MartainInGreen](https://github.com/MartianInGreen)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MartianInGreen/OpenWebUI-Tools&type=Date)](https://star-history.com/#MartianInGreen/OpenWebUI-Tools&Date)

If you want to support me check my [GitHub Profile](https://github.com/MartianInGreen) for more info. [Buy Me a Coffee](https://rennersh.de/buy-me-a-coffee)
