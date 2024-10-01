"""
title: Scrape Website
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
"""

import requests
import urllib, requests, os
from pydantic import BaseModel, Field #type: ignore
from typing import Callable, Awaitable


class Tools:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="",
            description="Base URL for the API.",
        )
        API_AUTH: str = Field(
            default="",
            description="API authentication"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True
        
    async def python_interpreter(self, uuid: str = "", code: str = ""):
        """
        Use a Python interpreter with internet access to execute code. No Notebook, use print etc. to output to STDOUT. Installed Libraries: numpy, scipy, pypdf2, pandas, pyarrow, matplotlib, pillow, opencv-python-headless, requests, bs4, geopandas, geopy, yfinance, seaborn, openpyxl, litellm, replicate, openai, ipython. Installed System libraries: wget git curl ffmpeg. You can link to files within the python intrpreter by using !(file_name)[https://api.rennersh.de/api/v1/interpreter/file/download/[uuid]/[filename]]. ALWAYS list the files before saying "can you upload that" or something similar, if the user is asking you to do something to a file they probably already uploaded it! You should use the same UUID for the entire conversation, unless the user specifically requests or gives you a new one.
        
        :param uuid: The UUID of the Python interpreter. If the user did not give you one, generate a new one. 
        :param code: The code to be executed. Formatted without any additional code blocks as a string.
        :return: The STD-OUT and STD-ERR of the executed python code.
        """
        endpoint = "/interpreter/python"
        url = f"{self.valves.BASE_URL}{endpoint}"
        payload = {
            "uuid": uuid,
            "code": code
        }
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.valves.API_AUTH}'}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    async def list_files(self, uuid: str, dirname: str = None):
        """
        List files in the interpreter.

        :param uuid: The UUID of the Python interpreter. If the user did not give you one, generate a new one. 
        :param dirname: Specify a directory to list files from. Can be left empty.
        :return: A list of files in the interpreter.
        """
        endpoint = "/interpreter/file/list"
        url = f"{self.valves.BASE_URL}{endpoint}"
        payload = {"uuid": uuid}
        if dirname:
            payload["dirname"] = dirname

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.valves.API_AUTH}'}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    async def create_session(self):
        """
        Create a new interpreter session. If you create a new one you need to provide the UUID you just generated to the user.

        :return: The UUID of the new interpreter session.
        """
        endpoint = "/interpreter/create"
        url = f"{self.valves.BASE_URL}{endpoint}"
        payload = {"create": "true"}
        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.valves.API_AUTH}'}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()