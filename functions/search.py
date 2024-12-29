"""
title: OpenWebUI Search
author: MartianInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
description: Advanced search tool for OpenWebUI.
required_open_webui_version: 0.3.30
requirements: langchain-openai==0.1.24, langgraph, aiohttp
version: 1.0.6
licence: MIT
"""

import os
import re
import time
import datetime
import json
from typing import (
    Callable,
    AsyncGenerator,
    Awaitable,
    Optional,
    Protocol,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

import inspect


# ---

import requests
import urllib
import aiohttp

# ---------------------------------------------------------------

# ---------------------------------------------------------------

EmitterType = Optional[Callable[[dict], Awaitable[None]]]


class SendCitationType(Protocol):
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


class SendStatusType(Protocol):
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


def get_send_citation(__event_emitter__: EmitterType) -> SendCitationType:
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            }
        )

    return send_citation


def get_send_status(__event_emitter__: EmitterType) -> SendStatusType:
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status


class Pipe:
    class Valves(BaseModel):
        OPENAI_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="Base URL for OpenAI API endpoints",
        )
        OPENAI_API_KEY: str = Field(default="", description="Primary API key")
        BRAVE_SEARCH_KEY: str = Field(default="", description="Brave Search API key")
        WOLFRAM_APPID: str = Field(default="", description="Wolfram Alpha AppID")
        PYTHON_BASE_URL: str = Field(default="", description="Base URL for the API.")
        PYTHON_API_AUTH: str = Field(default="", description="API authentication")
        ORCHESTRATOR_MODEL = Field(default="openai/gpt-4o-mini")
        SUMMARY_MODEL = Field(default="google/gemini-1.5-flash")
        ANSWER_MODEL = Field(default="openai/gpt-4o")
        AGENT_NAME: str = Field(default="Search", description="Name of the agent")
        AGENT_ID: str = Field(default="openwebui-search", description="ID of the agent")

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        os.environ["BRAVE_SEARCH_TOKEN"] = self.valves.BRAVE_SEARCH_KEY
        os.environ["BRAVE_SEARCH_TOKEN_SECONDARY"] = self.valves.BRAVE_SEARCH_KEY
        print(f"{self.valves=}")

    def pipes(self) -> list[dict[str, str]]:
        try:
            self.setup()
        except Exception as e:
            return [{"id": "error", "name": f"Error: {e}"}]

        return [{"id": self.valves.AGENT_ID, "name": self.valves.AGENT_NAME}]

    def setup(self):
        v = self.valves
        if not v.OPENAI_API_KEY or not v.OPENAI_BASE_URL:
            raise Exception("Error: OPENAI_API_KEY or OPENAI_BASE_URL is not set")
        self.openai_kwargs = {
            "base_url": v.OPENAI_BASE_URL,
            "api_key": v.OPENAI_API_KEY,
        }