"""
title: ReAct Toolchain, updated for newer version of OpenWebUI
author: MartianInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
description: ReAct is a toolchain agent with automatic model and tool selection. 
required_open_webui_version: 0.5.0
requirements: langchain-openai==0.2.14, langgraph==0.2.60, aiohttp
version: 2.0
licence: MIT
"""

from pydantic import BaseModel, Field, create_model
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent