# ----------------------------------------------------
# Developer: Hannah R. 
# Created: March 4th, 2025
# Lisence: MIT
# Description: Image Generation API Component
# ----------------------------------------------------

import aiohttp
import json, uuid
import base64
from io import BytesIO
from dotenv import load_dotenv
import os

# Load EXTERNAL_URL from .env file 
load_dotenv()
EXTERNAL_URL = os.getenv("EXTERNAL_URL")
BASE_URL= "https://fal.run"

async def create_image_basic(prompt: str, image_size: str = "square_hd", API_AUTH = ""):
    """
    Generates an image based on a given prompt using the basic endpoint.

    :param prompt: Detailed description of the image to generate.
    :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
    :return: URL to the generated image.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Key {API_AUTH}",
    }

    url = f"{BASE_URL}/fal-ai/flux/dev"
    payload = {"prompt": prompt, "image_size": image_size, "sync_mode": True}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            image_data_url = result["images"][0]["url"]
            
            # Save image to api/content

            id = uuid.uuid4()

            with open(f"api/content/{id}.png", "wb") as f:
                f.write(base64.b64decode(image_data_url.split(",")[1]))

            return {"result": + EXTERNAL_URL + str(id) + ".png"}
            
async def create_image_pro(prompt: str, image_size: str = "square_hd", API_AUTH = ""):
    """
    Generates an image based on a given prompt using the pro endpoint.

    :param prompt: Detailed description of the image to generate.
    :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
    :return: URL to the generated image.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Key {API_AUTH}",
    }

    url = f"{BASE_URL}/fal-ai/flux-pro/v1.1"
    payload = {"prompt": prompt, "image_size": image_size, "sync_mode": True}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            image_data_url = result["images"][0]["url"]
            
            # Save image to api/content

            id = uuid.uuid4()

            with open(f"api/content/{id}.png", "wb") as f:
                f.write(base64.b64decode(image_data_url.split(",")[1]))

            return {"result": + EXTERNAL_URL + str(id) + ".png"}

async def image_to_image(prompt: str, image_url: str = "", image_size: str = "", API_AUTH = ""):
    """
    Generates an image based on a given prompt using the image to image endpoint.

    :param prompt: Detailed description of the image to generate.
    :param image_url: URL to the image to use as a reference.
    :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
    :return: URL to the generated image.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Key {API_AUTH}",
    }

    url = f"{BASE_URL}/fal-ai/flux/dev/image-to-image"
    payload = {"prompt": prompt, "image_url": image_url, "image_size": image_size, "sync_mode": True}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            result = await response.json()
            image_data_url = result["images"][0]["url"]

            # Save image to api/content
            id = uuid.uuid4()

            with open(f"api/content/{id}.png", "wb") as f:
                f.write(base64.b64decode(image_data_url.split(",")[1]))

            return {"result": EXTERNAL_URL + str(id) + ".png"}