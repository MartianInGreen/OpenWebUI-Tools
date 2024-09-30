"""
title: Scrape Website
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
"""

import urllib, requests, os
from pydantic import BaseModel, Field #type: ignore
from typing import Callable, Awaitable


class Tools:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    def scrape_website(
        self, url: str, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> str:
        """
        Scrape any website and get redable markdown formatted results.
        :param url: The full url of the website you're trying to scrape. For example https://openai.com
        :return: Get an LLM optimzed Markdown version of the website.
        """
        try:
            # baseURL = f"http://api.wolframalpha.com/v2/query?appid={getEnvVar('WOLFRAM_APP_ID')}&output=json&input="
            baseURL = f"https://r.jina.ai/{url}"

            # Encode the query
            # encoded_query = urllib.parse.quote(url)
            url = baseURL  # + encoded_query

            try:
                response = requests.get(url)
                # print(response)
                data = response.text

                data = (
                    data
                    + "\n\n--------------------------------------\nSource Guidlines: Include a link to https://r.jina.ai/{website_url} in your response."
                )

                return data

            except Exception as e:
                # print(e)
                return "Error fetching Website results."
        except Exception as e:
            # print(e)
            return "Error fetching Website results."
