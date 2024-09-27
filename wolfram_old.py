"""
title: WolframAlpha API
author: ex0dus
author_url: https://github.com/roryeckel/open-webui-wolframalpha-tool
version: 0.2.0
"""

import os
import requests
import urllib.parse
from pydantic import BaseModel, Field
from typing import Callable, Awaitable


async def query_simple(
    query_string: str, app_id: str, __event_emitter__: Callable[[dict], Awaitable[None]]
) -> None:
    base_url = "http://api.wolframalpha.com/v1/simple"
    params = {"i": query_string, "appid": app_id}

    result_url = f"{base_url}?{urllib.parse.urlencode(params)}"

    await __event_emitter__(
        {
            "type": "message",
            "data": {"content": f"![WolframAlpha Simple Result]({result_url})"},
        }
    )


async def query_short_answer(
    query_string: str, app_id: str, __event_emitter__: Callable[[dict], Awaitable[None]]
) -> str:
    base_url = "http://api.wolframalpha.com/v1/result"
    params = {
        "i": query_string,
        "appid": app_id,
        "format": "plaintext",
    }

    await __event_emitter__(
        {
            "data": {
                "description": f"Performing WolframAlpha short answer query: {query_string}",
                "status": "in_progress",
                "done": False,
            },
            "type": "status",
        }
    )

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        text_response = response.text

        await __event_emitter__(
            {
                "data": {
                    "description": f"WolframAlpha returned: {text_response}",
                    "status": "complete",
                    "done": True,
                },
                "type": "status",
            }
        )
        return "WolframAlpha: " + text_response
    except Exception as e:
        print(e)
        await __event_emitter__(
            {
                "data": {
                    "description": f"Error: WolframAlpha returned {e}",
                    "status": "complete",
                    "done": True,
                },
                "type": "status",
            }
        )
        return f"There was an error fetching WolframAlpha response. You are required to report the following message to the user: {str(e)}"


class Tools:
    class Valves(BaseModel):
        WOLFRAMALPHA_APP_ID: str = Field(
            default="",
            description="The App ID (api key) to authorize WolframAlpha",
        )
        ENABLE_SIMPLE_API: bool = Field(
            default=True,
            description="Specify if the query should use the simple API. This will return images from WolframAlpha. Can be used in combination with short answer.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    # def get_app_id(self) -> str:
    #     """
    #     Get the App ID of the WolframAlpha query engine. This App ID is used to authenticate with WolframAlpha.
    #     :return: The App ID which is usually several characters split by a dash
    #     """
    #     return os.getenv("WOLFRAMALPHA_APP_ID")

    async def perform_query(
        self, query_string: str, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> str:
        """
        Query the WolframAlpha knowledge engine to answer a wide variety of complex mathematical formulas including trigonometry and differential equations. The engine also supports textual queries stated in English about other topics. You should cite this tool when it is used. It can also be used to supplement and back up knowledge you already know. WolframAlpha can also proive accurate real-time and scientific data (for example for elements, cities, weather, planets, etc. etc.)
        :param query_string: The question or mathematical equation to ask the WolframAlpha engine. DO NOT use backticks or markdown when writing your JSON request.
        :return: A short answer or explanation of the result of the query_string
        """
        app_id = self.valves.WOLFRAMALPHA_APP_ID or os.getenv("WOLFRAMALPHA_APP_ID")
        print(f"App ID = {app_id}")
        if not app_id:
            await __event_emitter__(
                {
                    "data": {
                        "description": f"Error: WolframAlpha APP_ID is not set",
                        "status": "complete",
                        "done": True,
                    },
                    "type": "status",
                }
            )
            return "You are required to report the following error message to the user: App ID is not set in the Valves or the environment variable 'WOLFRAMALPHA_APP_ID'."

        short_answer = await query_short_answer(query_string, app_id, __event_emitter__)

        if self.valves.ENABLE_SIMPLE_API:
            await query_simple(query_string, app_id, __event_emitter__)

        return short_answer
