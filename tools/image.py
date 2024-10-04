"""
title: Python Interpreter
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
requirements: aiohttp
"""

import aiohttp
import json, uuid
from pydantic import BaseModel, Field
from typing import Callable, Awaitable
import base64
from io import BytesIO

class Tools:
    class Valves(BaseModel):
        BASE_URL: str = Field(
            default="...",
            description="Base URL for the API.",
        )
        API_AUTH: str = Field(default="", description="API authentication")
        FAL_AI_KEY: str = Field(default="", description="API authentication")

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    async def create_image_basic(self, prompt: str, image_size: str = "square_hd"):
        """
        Generates an image based on a given prompt using the basic endpoint.

        :param prompt: Detailed description of the image to generate.
        :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
        :return: URL to the generated image.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Key {self.valves.API_AUTH}",
        }

        url = f"{self.valves.BASE_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "auth_key": self.valves.FAL_AI_KEY, "model": "dev"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result

    async def create_image_pro(self, prompt: str, image_size: str = "square_hd"):
        """
        Generates an image based on a given prompt using the pro endpoint.

        :param prompt: Detailed description of the image to generate.
        :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
        :return: URL to the generated image.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Key {self.valves.API_AUTH}",
        }

        url = f"{self.valves.BASE_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "auth_key": self.valves.FAL_AI_KEY, "model": "pro"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result

    async def image_to_image(self, prompt: str, image_url: str = "", image_size: str = "square_hd"):
        """
        Generates an image based on a given prompt using the image to image endpoint.

        :param prompt: Detailed description of the image to generate.
        :param image_url: URL to the image to use as a reference.
        :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
        :return: URL to the generated image.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Key {self.valves.API_AUTH}",
        }

        url = f"{self.valves.BASE_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "image_url": image_url, "auth_key": self.valves.FAL_AI_KEY, "model": "image-to-image"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result

