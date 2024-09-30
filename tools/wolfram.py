"""
title: WolframAlpha API
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
"""

import urllib, requests, os
from pydantic import BaseModel, Field #type: ignore
from typing import Callable, Awaitable

class Tools:
    class Valves(BaseModel):
        WOLFRAMALPHA_APP_ID: str = Field(
            default="",
            description="The App ID (api key) to authorize WolframAlpha",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    def wolframAlpha(
        self, query: str, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> str:
        """
        Query the WolframAlpha knowledge engine to answer a wide variety of questions. These questions can include real-time data questions, mathematical equasions or function, or scientific (data) questions. The engine also supports textual queries stated in English about other topics. You should cite this tool when it is used. It can also be used to supplement and back up knowledge you already know. WolframAlpha can also proive accurate real-time and scientific data (for example for elements, cities, weather, planets, etc. etc.)
        :param query: The question or mathematical equation to ask the WolframAlpha engine. DO NOT use backticks or markdown when writing your JSON request.
        :return: A short answer or explanation of the result of the query_string
        """
        try:
            # baseURL = f"http://api.wolframalpha.com/v2/query?appid={getEnvVar('WOLFRAM_APP_ID')}&output=json&input="
            baseURL = f"https://www.wolframalpha.com/api/v1/llm-api?appid={self.valves.WOLFRAMALPHA_APP_ID}&input="

            # Encode the query
            encoded_query = urllib.parse.quote(query)
            url = baseURL + encoded_query

            try:
                response = requests.get(url)
                #print(response)
                data = response.text

                data = data + "\nAlways include the Wolfram|Alpha website link in your response to the user!\n\nIf there are any images provided, think about displaying them to the user."
                
                return data

            except Exception as e:
                #print(e)
                return 'Error fetching Wolfram|Alpha results.'
        except Exception as e:
            #print(e)
            return 'Error fetching Wolfram|Alpha results.'