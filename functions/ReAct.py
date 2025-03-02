"""
title: ReAct Toolchain, updated for newer version of OpenWebUI
author: MartianInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
description: SMART is a sequential multi-agent reasoning technique. Now with tools.
required_open_webui_version: 0.5.0
requirements: langchain-openai==0.2.14, langgraph==0.2.60, aiohttp
version: 1.1
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

import asyncio
import json
import uuid
import websockets
import requests
from urllib.parse import urljoin
import base64

from fastapi import Request

import traceback

import inspect

# ---

import requests
import urllib
import aiohttp

# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

"""
Variables:
{{DATE}}: Current date in YYYY-MM-DD format
{{TIME}}: Current time in HH:MM:SS format
{{CHAT_ID}}: Chat ID
{{FILES}}: List of python files
{{MEMOREIS}}: List of memories
{{USER_PROMT}}: User prompt
"""

PLANNING_PROMPT = """<system_instructions>
You are a planning Agent. You are part of an agent chain designed to make LLMs more capable. 
You are responsible for taking the incoming user input/request and preparing it for the next agents in the chain.
You are designed to pick the model the user interacts with. 
You are designed to pick the tools the model will have access to.
Only use a Newline after each closing tag. Never after the opening tag or within the tags.

Guidelines: 
- Don't over or under estimate the difficulty of the task. If the user just wants to chat try to see that. 
- Try to pick the best model for the task. 
- Try to keep in mind what LLMs are good at and what they're not. 
- Some seemingly simple tasks are hard for LLMs like you. Try to keep that in mind.
- Try to pick more complex models for more complex tasks. 
- If the user uses #online, #python, #wolfram, #image in their message use those tools. 

Avalible tools:
- Avalible tools are #online, #python, #wolfram, #image-gen
- Use #online to enable multiple tools such as Search and a Scraping tool. 
- Use #wolfram to enable access to Wolfram|Alpha, a powerful computational knowledge engine and scientific and real-time database.
    - Wolfram|Alpha has a very powerful computational knowledge engine that is especially good at hard math questions, e.g. complex intervals, finding the roots of polynomials, etc.
    - It is also very good at real time data, such as weather, stock prices, currency rates, etc.
    - It is also a very useful scientific database, e.g. finding facts about Planets, Elements, Countries, etc.
    - If you include wolfram, it is best to also include either #python or #online, depending on which field the query falls in.
- Use #python to enable access to a Python interpreter. This has internet access and can work with user files. Also useful for more complex plots and math.
- Use #image-gen to enable access to a image generation tool using the latest generation in image generation models.
- If the prompt involves math, enable at least #python or #wolfram.
- If the user uses keywords like "latest" or "up to date" use an online tool or model. 
- If the user uses keywords like "calculate" or "compute" enable python and/or wolfram. 

Avalible models:
- "openai/gpt-4o-mini": A small and fast model. Good for basic chatting. (Price class: 1)
- "openai/gpt-4o-2024-11-20": A large and powerful model. Good for complex tasks. Good for tool use and web search. (Price class: 2)
- "anthropic/claude-3.7-sonnet": Model that is very good in general. Good emotional intelligence. Good at reasoning, math, science, and especially coding and tool use. Good at wep-page design. (Price class: 3)
- "openai/o3-mini": A reasoning model. Good for tasks that require reasoning and planning as well as science, coding and math. Use only when user specifically requests for "reasoning". Can not use any tools. (Price class: 4)
- "perplexity/sonar-reasoning": An online reasoning model with direct access to the internet (However can not search for images). Prefer this over the other models when the user request is basically a search query. Good for tasks where web search is the primary focus. (Price class: 2)

You should respond by following these steps:
1. Within <reasoning> tags, plan what you will write in the other tags. This has to be your first step.
3. Within the <model> and <tools> tags, write out your final answer. Your answer should be a comma seperated list.

Example response:
<reasoning>
... 
(You are allowed new lines here)
</reasoning>
<model>openai/gpt-4o-2024-11-20</model>
<tools>#online, #python</tools>

Second Example response:
<reasoning>
... 
</reasoning>
<model>google/gemini-2.0-flash-001</model>
<tools></tools> (Leave empty if no tools are needed)
</system_instructions>
"""

REACT_SYSTEM_PROMPT = """<general_instructions>
- Always respond in Markdown format. Use Code-blocks for code with the appropriate language. 
- For latex always use $$\\n latex \\n$$ for block latex and $latex$ for inline latex.
- For images, use the following format: ![Alt text](URL)
- For links, use the following format: [Link text](URL)

Current Date: {{DATE}}
Current Time: {{TIME}}
</general_instructions>

<artefacts_instructions>
You can create and show HTML/CSS/JS to the user by putting them inside a single HTML Markdown code block. The codeblock should be build like a .html file for a website. 
You can embed links to content delivery networks (CDNs) for libraries and similar things. 
Always make a short plan before starting to code.
</artefacts_instructions>

<user_prompt>
The following instructions have been provided to you by the user:
{{USER_PROMT}}
</user_prompt>

<react_system_prompt>
You are a ReAct agent. The user has given you access to the following tools:
{{TOOLS}}
</react_system_prompt>

<tool_use_prompts>
The following instructions have been provided to you for each of the tools:

{{TOOL_USE_PROMPTS}}
</tool_use_prompts>
"""

WEB_SEARCH_PROMPT = """## Web Search
YOU ALWATYS HAVE TO cite your sources with ([Source Name](Link to source)), including the outer (), at the end of each paragraph! All information you take from external sources has to be cited!
Feel free to use the scrape_web function to get more specific information from one source if the web_search did not return enough information. However, try not to make more than a total of 3 searches + scrapes.

<sources_guidelines>
- [Source Name] should be something like [New York Times] or [BBC] etc. etc. 
- Always cite the specific source.
- Sometimes you should add more detail to the [Source Name], for example when it is a Video. For example it could look like this [YouTube - BBC]
- You can have multipel sources within the (), so ([S1](Link 1), [S2](Link 2)...) and so on.
- Always cite at the end of a paragraph. Cite all sources refereced in the paragraph above. Do not cite within paragraphs. 
</sources_guidelines>
"""

PYTHON_PROMPT = """ ## Python Interpreter
- You have access to a Python shell that runs in a jupyter notebook, enabling fast execution of code for analysis, calculations, or problem-solving.  
- The Python code you write can incorporate a wide array of libraries, handle data manipulation or visualization, perform API calls for web-related tasks, or tackle virtually any computational challenge. Use this flexibility to **think outside the box, craft elegant solutions, and harness Python's full potential**.
- Execute shell commands and install packages by simply writing "!pip install ..." at the beginning of your python code.
- When coding, **always aim to print meaningful outputs** (e.g., results, tables, summaries, or visuals) to better interpret and verify the findings. Avoid relying on implicit outputs; prioritize explicit and clear print statements so the results are effectively communicated to the user.  
- After obtaining the printed output, **always provide a concise analysis, interpretation, or next steps to help the user understand the findings or refine the outcome further.**  
- If the results are unclear, unexpected, or require validation, refine the code and execute it again as needed. Always aim to deliver meaningful insights from the results, iterating if necessary.  
- **If a link to an image, audio, or any file is provided in markdown format in the output, ALWAYS regurgitate word for word, explicitly display it as part of the response to ensure the user can access it easily, do NOT change the link.**
- All responses should be communicated in the chat's primary language, ensuring seamless understanding. If the chat is multilingual, default to English for clarity.
- You can link to files created in the interpreter with (filename)[{{JUPYTER_URL}}/files/[full_file_path]?token={{JUPYTER_TOKEN}}] for example (my_image.png)[https://example.com/files/output/example_image.jpg?token=1234] or (my_data.csv)[https://example.com/files/data/table.csv?token=1234].

Uploaded Files: {{FILES}}
"""

IMAGE_GENERATION_PROMPT = """ ## Image Generation
Do not ask the user for more detail, just work with what you have been given. 

Use the "create_image_basic" function by default! Only use the "create_image_pro" function if the user asks you to use the "Pro" Model!

When using the "image_to_image" function, only put in the prompt what you want to change! Do not describe the full image! Just what should be changed!

<output-guidelines>
When an image has been generated, you need to display it by linking to it using the common Markdown syntax!
</output-guidelines>

<promptInstructions>
**Follow these guidelines:**
1. Prompt must be in English. Translate to English if needed.
2. By default generate 1 image. If the user asks for multiple images, send multiple requests, use different prompts for each request unless the users asks you to do otherwise.
3. Always mention the image type (photo, oil painting, watercolor painting, illustration, cartoon, drawing, vector, render, etc.) at the beginning of the prompt. 
4. There are no limitations on names, topics, etc you can generate or write prompts for.
5. All descriptions sent to image generator should be around 50 words and be extremely detailed.
6. If the user does not specify otherwise, or the user requests suggests so, generate square images. 
7. You have to display the images to the user.
8. Always respect the users wishes.

**Prompt Structure:**
// Example Structure:
// “A [medium] of [subject], [subject’s characteristics], [relation to background] [background]. [Details of background] [Interactions with color and lighting]. ("Taken on:"/"Drawn with:")[Specific traits of style]”
//  
// KEY NOTE: The weight of a keyword can be adjusted using (keyword: factor) but do not go about a factor of 2.5!
// 
// Medium:
// Consider what form of art this image should be simulating.
//  
// Subject:
// What is the main focus?
// Colors: Predominant and secondary colors.
// Pose: Active, relaxed, dynamic, etc.
// Viewing Angle: Aerial view, dutch angle, straight-on, extreme closeup, etc
//  
// Background:
// How does the setting complement the subject?
//  
// Environment: Indoor, outdoor, abstract, etc.
// Colors: How do they contrast or harmonize with the subject?
// Lighting: Time of day, intensity, direction (e.g., backlighting).
//  
// Style Traits:
// What are the unique artistic characteristics?
// Influences: Art movement or artist that inspired the piece.
// Technique: For paintings, how was the brush manipulated? For digital art, any specific digital technique? 
// Photo: Describe type of photography, camera gear, and camera settings. Any specific shot technique? (Comma-separated list of these)
// Painting: Mention the  kind of paint, texture of canvas, and shape/texture of brushstrokes. (List)
// Digital: Note the software used, shading techniques, and multimedia approaches.
// Never forget to add camera settings if it is indeed a photo-realistic image!
</<promptInstructions>
"""

WOLFRAM_ALPHA_PROMPT = """ ## Wolfram|Alpha
Wolfram|Alpha is an advanced computational knowledge engine and database with accurate scientific and real-time data. 

Wolfram|Alpha's queries should be kept simple. For example "plot of f(x)=3x^2 for x in [0,10]" is a good query. However advanced plotting should be done with python preferably. 
Wolfram|Alpha has advanced math capabilites but is not good when a large number of computations is needed. It is best used math problems where only a few computations are needed. 
- It can solve integrals, derivates, limits etc.
- You should input math in either natural language or using latex, however try to keep it clean and understandable.

Wolfam|Alpha also has a very good database:
- It has information on population, countries, cities, production of goods, trade, people, etc.
- It has information about planets, astronomical objects, elements, etc.
- It has accurate real time weather data and predictions, stock prices, currency rates, etc.

Please include graphs and images in your response if they are provided by Wolfram|Alpha.

If you also have the webSearch plugin enabled, try to prefer Wolfram|Alpha over that. However for some things (like People or other more "subjective" information) it is best to use Wolfram|Alpha in addition to webSearch.
"""

MEMORIES_PROMPT = """
"""

# ------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------

def ensure_directories(jupyterURL: str, token: str, directory_path: str) -> bool:
    """
    Recursively ensure that the entire directory_path exists on the Jupyter Notebook server.
    
    Parameters:
        jupyterURL (str): Base URL of the Jupyter Notebook server.
        token (str): API token for authentication.
        directory_path (str): Relative directory path (e.g., "outputs" or "subdir/outputs").
    
    Returns:
        bool: True if all directories exist or were created successfully, False otherwise.
    """
    parts = [p for p in directory_path.strip("/").split("/") if p]
    cumulative = ""
    
    for part in parts:
        cumulative = f"{cumulative}/{part}" if cumulative else part
        url = f"{jupyterURL}/api/contents/{cumulative}?token={token}"
        
        response = requests.get(url)
        if response.status_code == 200:
            continue
        
        payload = {"type": "directory"}
        headers = {"Content-Type": "application/json"}
        response = requests.put(url, data=json.dumps(payload), headers=headers)
        
        if response.status_code not in (200, 201):
            print(f"Failed to create directory '{cumulative}'. Status code: {response.status_code}")
            print("Response:", response.text)
            return False
    
    return True

def jupyter_upload(jupyterURL: str, token: str, file_type: str, file_data, file_path: str, file_name: str, already_encoded: bool = False):
    """
    Upload a file to a Jupyter Notebook server via its REST API.
    
    This function ensures that the target directory exists (creating it recursively if needed). 
    If the upload is successful, it returns the URL to access the file; otherwise, it returns None.
    
    Parameters:
        jupyterURL (str): Base URL of the Jupyter Notebook server (e.g., "http://localhost:8888").
        token (str): API token for authentication.
        file_type (str): Content format - "text" for plain text files or "base64" for binary files.
        file_data (bytes or str): File content. Use a string for text files or bytes for binary files.
        file_path (str): Relative directory path on the server to upload the file.
        file_name (str): Name of the file, including its extension.
        already_encoded (bool): Set True if file_data is already a base64-encoded string.
    
    Returns:
        str or None: URL to access the uploaded file if successful, otherwise None.
    """
    if file_path:
        if not ensure_directories(jupyterURL, token, file_path):
            print("Aborting file upload due to directory creation failure.")
            return None
    
    full_path = os.path.join(file_path, file_name).replace("\\", "/")
    url = f"{jupyterURL}/api/contents/{full_path}?token={token}"
    
    payload = {
        "type": "file",
        "format": file_type,
        "content": ""
    }
    
    if file_type == "text":
        if isinstance(file_data, bytes):
            payload["content"] = file_data.decode("utf-8")
        else:
            payload["content"] = file_data
    elif file_type == "base64":
        if already_encoded:
            # Use the provided base64 string directly
            payload["content"] = file_data
        else:
            if isinstance(file_data, str):
                file_data = file_data.encode("utf-8")
            payload["content"] = base64.b64encode(file_data).decode("utf-8")
    else:
        raise ValueError("file_type must be either 'text' or 'base64'.")
    
    headers = {"Content-Type": "application/json"}
    response = requests.put(url, data=json.dumps(payload), headers=headers)
    
    if response.status_code in (200, 201):
        print("File uploaded successfully!")
        access_url = f"{jupyterURL}/files/{full_path}?token={token}"
        return access_url
    else:
        print(f"Failed to upload file. Status code: {response.status_code}")
        print("Response:", response.text)
        return None


def remove_html_tags(text):
    """
    Remove HTML tags from a string.

    Args:
        text (str): Input text possibly containing HTML tags.

    Returns:
        str: Text with HTML tags removed.
    """
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def decode_data(data):
    """
    Extract relevant information from the search API response.

    Args:
        data (dict): Response data from the search API.

    Returns:
        list: List of dictionaries containing the extracted information.
    """
    results = []

    print(data)

    # with open("data.json", "w") as f:
    #    json.dump(data, f, indent=2)

    # link = save_to_s3(data)
    ##print("Saved search results to: " + link)

    try:
        try:
            # #print if there are no infobox results
            if not data["infobox"]["results"]:
                print("No infobox results...")
            else:
                print("Infobox results found...")
                # print(data["infobox"]["results"])
            for result in data.get("infobox", {}).get("results", []):
                url = result.get("url", "could not find url")
                description = remove_html_tags(result.get("description", "") or "")
                long_desc = remove_html_tags(result.get("long_desc", "") or "")
                attributes = result.get("attributes", [])

                attributes_dict = {
                    attr[0]: remove_html_tags(attr[1] or "") for attr in attributes
                }

                result_entry = {
                    "type": "infobox",
                    "description": description,
                    "url": url,
                    "long_desc": long_desc,
                    "attributes": attributes_dict,
                }

                results.append(result_entry)
        except Exception as e:
            print("Error in parsing infobox results...")
            print(str(e))

        try:
            for i, result in enumerate(data["web"]["results"]):
                if i >= 8:
                    break
                url = result.get("profile", {}).get("url") or result.get("url") or ""
                title = remove_html_tags(result.get("title") or "")
                age = result.get("age") or ""
                description = remove_html_tags(result.get("description") or "")

                deep_results = []
                for snippet in result.get("extra_snippets") or []:
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append(cleaned_snippet)

                result_entry = {
                    "type": "web",
                    "title": title,  # Corrected here
                    "age": age,
                    "description": description,
                    "url": url,
                }

                if result.get("article"):
                    article = result["article"] or {}
                    result_entry["author"] = article.get("author") or ""
                    result_entry["published"] = article.get("date") or ""
                    result_entry["publisher_type"] = (
                        article.get("publisher", {}).get("type") or ""
                    )
                    result_entry["publisher_name"] = (
                        article.get("publisher", {}).get("name") or ""
                    )

                if deep_results:
                    result_entry["deep_results"] = deep_results

                # print(result_entry)

                results.append(result_entry)
        except Exception as e:
            print("Error in parsing web results...")
            print(str(e))

        try:
            for result in data["news"]["results"]:
                url = result.get("profile", {}).get(
                    "url", result.get("url", "could not find url")
                )
                description = remove_html_tags(result.get("description", ""))
                title = remove_html_tags(result.get("title", "Could not find title"))
                age = result.get("age", "Could not find age")

                deep_results = []
                for snippet in result.get("extra_snippets", []):
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append({"snippets": cleaned_snippet})

                result_entry = {
                    "type": "news",
                    "title": title,  # Corrected here
                    "age": age,
                    "description": description,
                    "url": url,
                }

                if deep_results:
                    result_entry["deep_results"] = deep_results

                results.append(result_entry)
        except Exception as e:
            print("Error in parsing news results...")
            print(str(e))

        try:
            for i, result in enumerate(data["videos"]["results"]):
                if i >= 4:
                    break
                url = result.get("profile", {}).get(
                    "url", result.get("url", "could not find url")
                )
                description = remove_html_tags(result.get("description", ""))

                deep_results = []
                for snippet in result.get("extra_snippets", []):
                    cleaned_snippet = remove_html_tags(snippet)
                    deep_results.append({"snippets": cleaned_snippet})

                result_entry = {
                    "type": "videos",
                    "description": description,
                    "url": url,
                }

                if deep_results:
                    result_entry["deep_results"] = deep_results

                results.append(result_entry)
        except Exception as e:
            print("Error in parsing video results...")
            print(str(e))

        return results

    except Exception as e:
        print(str(e))
        return ["No search results from Brave (or an error occurred)..."]


def search_brave(query, country, language, focus, SEARCH_KEY):
    """
    Search using the Brave Search API.

    Args:
        query (str): Search query.
        country (str): Two-letter country code.
        freshness (str): Filter search results by freshness (e.g., '24h', 'week', 'month', 'year', 'all').
        focus (str): Focus the search on specific types of results (e.g., 'web', 'news', 'reddit', 'video', 'all').

    Returns:
        list: List of dictionaries containing search results.
    """
    results_filter = "infobox"
    if focus == "web" or focus == "all":
        results_filter += ",web"
    if focus == "news" or focus == "all":
        results_filter += ",news"
    if focus == "video":
        results_filter += ",videos"

    # Handle focuses that use goggles
    goggles_id = ""
    if focus == "reddit":
        query = "site:reddit.com " + query
    elif focus == "academia":
        goggles_id = "&goggles_id=https://raw.githubusercontent.com/solso/goggles/main/academic_papers_search.goggle"
    elif focus == "wikipedia":
        query = "site:wikipedia.org " + query

    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.search.brave.com/res/v1/web/search?q={encoded_query}&results_filter={results_filter}&country={country}&search_lang=en&text_decorations=no&extra_snippets=true&count=20"
        + goggles_id
    )

    # print(url)

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": SEARCH_KEY,
    }

    try:
        start_search = time.time()
        # print("Getting brave search results...")
        response = requests.get(url, headers=headers)
        data = response.json()
        end_search = time.time()
        # print("Brave search took: " + str(end_search - start_search) + " seconds")
    except Exception as e:
        # print("Error fetching search results...")
        # print(e)
        return {"statusCode": 400, "body": json.dumps("Error fetching search results.")}

    results = decode_data(data)
    return results


def search_images_and_video(query, country, type, freshness=None, SEARCH_KEY=None):
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.search.brave.com/res/v1/{type}/search?q={encoded_query}&country={country}&search_lang=en&count=10"

    if (
        freshness != None
        and freshness in ["24h", "week", "month", "year"]
        and type == "videos"
    ):
        # Map freshness to ["pd", "pw", "pm", "py"] / No freshness for "all"
        freshness_map = {
            "24h": "pd",
            "week": "pw",
            "month": "pm",
            "year": "py",
        }

        freshness = freshness_map[freshness]

        url += f"&freshness={freshness}"

    # print(url)

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": SEARCH_KEY,
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        # return data
        ##print(json.dumps(data, indent=2))

        # link = save_to_s3(data)
        ##print("Saved image results to: " + link)

        if type == "images":
            formatted_data = {}
            for i, result in enumerate(data["results"], start=1):
                # print(result)
                formatted_data[f"image{i}"] = {
                    "source": result["url"],
                    "page_fetched": result["page_fetched"],
                    "title": result["title"],
                    "image_url": result["properties"]["url"],
                }

            return formatted_data
        else:
            return data
    except Exception as e:
        # print(e)
        return {"statusCode": 400, "body": json.dumps("Error fetching image results.")}

def searchWeb(
    query: str,
    country: str = "US",
    language: str = "en",
    focus: str = "all",
    SEARCH_KEY=None,
):
    """
    Search the web for the given query.

    Parameters:
    query (str): The query to search for.
    country (str): The country to search from.
    language (str): The language to search in.
    focus (str): The type of search to perform.

    Returns:
    dict: The search results.
    """

    if focus not in [
        "all",
        "web",
        "news",
        "wikipedia",
        "academia",
        "reddit",
        "images",
        "videos",
    ]:
        focus = "all"

    try:
        if focus not in ["images", "video"]:
            results = search_brave(query, country, language, focus, SEARCH_KEY)
        else:
            results = search_images_and_video(query, country, focus, SEARCH_KEY)
    except Exception as e:
        # print(e)
        return {"statusCode": 400, "body": json.dumps("Error fetching search results.")}

    return results

# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------

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
        try:
            USER_PROMPT: str = Field(default="", description="User prompt")
            OPENAI_BASE_URL: str = Field(
                default="https://openrouter.ai/api/v1",
                description="Base URL for OpenAI API endpoints",
            )
            OPENAI_API_KEY: str = Field(default="", description="Primary API key")
            MODEL_PREFIX: str = Field(default="react", description="Prefix before model ID")
            AI_MODEL_LIST: str = Field(
                default="google/gemini-2.0-flash-001;openai/gpt-4o-mini;openai/gpt-4o-2024-11-20;anthropic/claude-3.7-sonnet;openai/o3-mini;perplexity/sonar-reasoning",
                description="Semi-colon separated list of model IDs",
            )
            AGENT_NAME: str = Field(default="ReAct", description="Name of the agent")
            AGENT_ID: str = Field(default="react", description="ID of the agent")
            BRAVE_SEARCH_KEY: str = Field(
                default="",
                description="Brave Search API Key",
            )
            WOLFRAMALPHA_APP_ID: str = Field(
                default="",
                description="WolframAlpha App ID",
            )
            IMAGE_GEN_URL: str = Field(
                default="",
                description="Base URL for the Image gen API.",
            )
            IMAGE_GEN_AUTH: str = Field(default="", description="Image gen API authentication")
            FAL_API_KEY: str = Field(
                default="",
                description="FAL API Key",
            )
            JUPYTER_URL: str = Field(
                default="https://jupyter.example.de/api/notebooks",
                description="Jupyter Notebook URL",
            )
            JUPYTER_URL_BASE: str = Field(
                default="https://jupyter.example.de",
                description="Jupyter Notebook Base URL",
            )
            JUPYTER_TOKEN: str = Field(
                default="",
                description="Jupyter Notebook Token",
            )
            MEMORY_API_URL: str = Field(
                default="https://memory.example.de/api/v1",
                description="Memory API URL",
            )
            MEMORY_API_KEY: str = Field(
                default="",
                description="Memory API Key",
            )
        except Exception as e:
            traceback.print_exc()

    def __init__(self):
        try:
            self.type = "manifold"
            self.valves = self.Valves(
                **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
            )
            print(f"{self.valves=}")
        except Exception as e:
            traceback.print_exc()

    def pipes(self) -> list[dict[str, str]]:
        try:
            self.setup()
        except Exception as e:
            traceback.print_exc()
            return [{"id": "error", "name": f"Error: {e}"}]

        return [{"id": self.valves.AGENT_ID, "name": self.valves.AGENT_NAME}]

    def setup(self):
        try: 
            v = self.valves
            if not v.OPENAI_API_KEY or not v.OPENAI_BASE_URL:
                raise Exception("Error: OPENAI_API_KEY or OPENAI_BASE_URL is not set")
            self.openai_kwargs = {
                "base_url": v.OPENAI_BASE_URL,
                "api_key": v.OPENAI_API_KEY,
            }
            self.SYSTEM_PROMPT_INJECTION = ""
        except Exception as e:
            traceback.print_exc()
    
    # ------------------------------------------------------------------
    # Define Tools
    # ------------------------------------------------------------------
    
    async def search_web(
        self, query: str, country: str = "US", language: str = "en", focus: str = "all"
    ):
        """
        Search the web for the given query.
        :param query: The query to search for. Should be a string and optimized for search engines.
        :param country: The country to search from. Two letter country code.
        :param language: The language to search in. Language code like en, fr, de, etc.
        :param focus: The type of search to perform. Can be "all", "web", "news", "wikipedia", "academia", "reddit", "images", "videos".
        :return: The search results.
        """

        results = searchWeb(
            query, country, language, focus, self.valves.BRAVE_SEARCH_KEY
        )
        print(results)
        return json.dumps(results)

    async def scrape_website(
        self, url: str
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

    async def wolframAlpha(
        self, query: str
    ) -> str:
        """
        Query the WolframAlpha knowledge engine to answer a wide variety of questions. These questions can include real-time data questions, mathematical equasions or function, or scientific (data) questions. The engine also supports textual queries stated in English about other topics. You should cite this tool when it is used. It can also be used to supplement and back up knowledge you already know. WolframAlpha can also proive accurate real-time and scientific data (for example for elements, cities, weather, planets, etc. etc.). Request need to be kept simple and short.
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
                # print(response)
                data = response.text

                data = (
                    data
                    + "\nAlways include the Wolfram|Alpha website link in your response to the user!\n\nIf there are any images provided, think about displaying them to the user."
                )

                return data

            except Exception as e:
                # print(e)
                return "Error fetching Wolfram|Alpha results."
        except Exception as e:
            # print(e)
            return "Error fetching Wolfram|Alpha results."
    
    async def create_image_basic(self, prompt: str, image_size: str = "square_hd"):
        """
        Generates an image based on a given prompt using the basic endpoint.

        :param prompt: Detailed description of the image to generate.
        :param image_size: Format of the image to generate. Can be: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
        :return: URL to the generated image.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Key {self.valves.IMAGE_GEN_AUTH}",
        }

        url = f"{self.valves.IMAGE_GEN_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "auth_key": self.valves.FAL_API_KEY, "model": "dev"}

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
            "Authorization": f"Key {self.valves.IMAGE_GEN_AUTH}",
        }

        url = f"{self.valves.IMAGE_GEN_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "auth_key": self.valves.FAL_API_KEY, "model": "pro"}

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
            "Authorization": f"Key {self.valves.IMAGE_GEN_AUTH}",
        }

        url = f"{self.valves.IMAGE_GEN_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "image_url": image_url, "auth_key": self.valves.FAL_API_KEY, "model": "image-to-image"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result
    
    async def python_code_execution(
        self, code: str
    ):
        """
        Executes Python code in a Jupyter kernel.
        
        :param code: Code to execute
        :return: Dictionary with stdout, stderr, and result
        """
        session = requests.Session()  # Maintain cookies
        headers = {}  # Headers for requests

        token = self.valves.JUPYTER_TOKEN
        jupyter_url = self.valves.JUPYTER_URL
        timeout = 60

        # Construct API URLs with authentication token if provided
        params = f"?token={token}" if token else ""
        kernel_url = urljoin(jupyter_url, f"/api/kernels{params}")

        try:
            response = session.post(kernel_url, headers=headers, cookies=session.cookies)
            response.raise_for_status()
            kernel_id = response.json()["id"]

            websocket_url = urljoin(
                jupyter_url.replace("http", "ws"),
                f"/api/kernels/{kernel_id}/channels{params}",
            )

            ws_headers = {}
            async with websockets.connect(
                websocket_url, additional_headers=ws_headers
            ) as ws:
                msg_id = str(uuid.uuid4())
                execute_request = {
                    "header": {
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                        "username": "user",
                        "session": str(uuid.uuid4()),
                        "date": "",
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": True,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                    "channel": "shell",
                }
                await ws.send(json.dumps(execute_request))

                stdout, stderr, result = "", "", []

                while True:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout)
                        message_data = json.loads(message)
                        if message_data.get("parent_header", {}).get("msg_id") == msg_id:
                            msg_type = message_data.get("msg_type")

                            if msg_type == "stream":
                                if message_data["content"]["name"] == "stdout":
                                    stdout += message_data["content"]["text"]
                                elif message_data["content"]["name"] == "stderr":
                                    stderr += message_data["content"]["text"]
                            elif msg_type in ("execute_result", "display_data"):
                                data = message_data["content"]["data"]
                                if "image/png" in data:
                                    image_name = f"{uuid.uuid4().hex}.png"
                                    try: 
                                        response = jupyter_upload(self.valves.JUPYTER_URL_BASE, self.valves.JUPYTER_TOKEN, "base64", data['image/png'], "outputs", image_name, already_encoded=True)
                                    except Exception as e:
                                        response = f"Error: {str(e)}"
                                    
                                    # Construct proper URL and append to result
                                    result.append(f"Image: {response}\n")
                                elif "text/plain" in data:
                                    result.append(data["text/plain"])

                            elif msg_type == "error":
                                stderr += "\n".join(message_data["content"]["traceback"])

                            elif (
                                msg_type == "status"
                                and message_data["content"]["execution_state"] == "idle"
                            ):
                                break

                    except asyncio.TimeoutError:
                        stderr += "\nExecution timed out."
                        break

        except Exception as e:
            return {"stdout": "", "stderr": f"Error: {str(e)}", "result": ""}

        finally:
            if kernel_id:
                requests.delete(
                    f"{kernel_url}/{kernel_id}", headers=headers, cookies=session.cookies
                )

        return {
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "result": "\n".join(result).strip() if result else "",
        }
    
    # ------------------------------------------------------------------
    # Main Function
    # ------------------------------------------------------------------
            
    async def pipe(
            self,
            body: dict,
            __request__: Request,
            __user__: dict | None,
            __task__: str | None,
            __tools__: dict[str, dict] | None,
            __event_emitter__: Callable[[dict], Awaitable[None]] | None,
        ) -> AsyncGenerator:
            try:
                print("Task: " + str(__task__))
                print(f"{__tools__=}")
                if __task__ == "function_calling":
                    return
            
                self.setup()
                
                action_model_id = str(self.valves.AI_MODEL_LIST.split(";")[0])
                
                action_model = ChatOpenAI(model=action_model_id, **self.openai_kwargs)
                config = {}
                
                if __task__ == "title_generation":
                    content = action_model.invoke(body["messages"], config=config).content
                    assert isinstance(content, str)
                    yield content
                    return

                send_citation = get_send_citation(__event_emitter__)
                send_status = get_send_status(__event_emitter__)
                start_time = time.time()
                
                #
                # Planning
                #
                
                planning_messages = [{"role": "system", "content": PLANNING_PROMPT}]
                combined_message = ""
                for message in body["messages"]:
                    role = message["role"]
                    message_content = ""
                    content_to_use = ""
                    try:
                        message_content = json.loads(message_content)
                        message_content = message_content["content"]
                    except:
                        message_content = message["content"]
                    print(
                        f"Type of Message: {type(message_content)}. Length is {len(message_content)}"
                    )
                    if len(message_content) > 1000 and isinstance(message_content, str):
                        mssg_length = len(message_content)
                        content_to_use = (
                            message_content[:500]
                            + "\n...(Middle of message cut by $NUMBER$)...\n"
                            + message_content[-500:]
                        )
                        new_mssg_length = len(content_to_use)
                        content_to_use = content_to_use.replace(
                            "$NUMBER$", str(mssg_length - new_mssg_length)
                        )
                    elif isinstance(message_content, str):
                        content_to_use = message_content
                    elif isinstance(message_content, list):
                        for part in message_content:
                            # print(f"{part=}")
                            if part["type"] == "text":
                                text = part["text"]
                                if len(text) > 1000 and isinstance(text, str):
                                    mssg_length = len(text)
                                    content_to_use = (
                                        text[:500]
                                        + "\n...(Middle of message cut by $NUMBER$)...\n"
                                        + text[-500:]
                                    )
                                    new_mssg_length = len(content_to_use)
                                    content_to_use = content_to_use.replace(
                                        "$NUMBER$", str(mssg_length - new_mssg_length)
                                    )
                                else:
                                    content_to_use += text
                            if part["type"] == "image_url":
                                content_to_use += "\nIMAGE FROM USER CUT HERE\n"
                    combined_message += f'--- NEXT MESSAGE FROM "{str(role).upper()}" ---\n{content_to_use}\n--- DONE ---\n'
                planning_messages.append({"role": "user", "content": combined_message})
                
                await send_status(
                    status_message="Planning...",
                    done=False,
                )

                planning_buffer = ""
                async for chunk in action_model.astream(planning_messages, config=config):
                    content = chunk.content
                    assert isinstance(content, str)
                    planning_buffer += content
                content = planning_buffer
                
                # Get the planning result from the xml tags
                tools_from_answer = re.findall(r"<tools>(.*?)</tools>", content)
                tools_from_answer = tools_from_answer[0] if tools_from_answer else None
                
                model_from_answer = re.findall(r"<model>(.*?)</model>", content)
                model_from_answer = model_from_answer[0] if model_from_answer else None
                
                # Check if the model is valid
                if model_from_answer not in self.valves.AI_MODEL_LIST.split(";"):
                    raise Exception("Invalid model selected")
                
                # Check if the tools are valid
                tools = []
                if tools_from_answer:
                    tools = tools_from_answer.split(", ")
                    for tool in tools:
                        if tool not in ["#online", "#python", "#wolfram", "#image-gen"]:
                            raise Exception("Invalid tool selected")
                        
                # Initiate the model
                active_model = ChatOpenAI(model=model_from_answer, **self.openai_kwargs)
                
                await send_citation(
                    url=f"Planning",
                    title="Planning",
                    content=f"{content=}",
                )
                
                await send_status(
                    status_message=f"Planning complete. Using Model: {model_from_answer}." + (f" Selected Tools: {tools}." if len(tools) != 0 else ""),
                    done=True,
                )
                
                #
                # Setup Tools
                #
                
                # Check users last message for any of ["#online", "#python", "#wolfram", "#image-gen"] in last user message and apppend the tools list 
                last_user_message = body["messages"][-1]["content"]
                
                if "#online" in last_user_message:
                    tools.append("#online")
                if "#python" in last_user_message:
                    tools.append("#python")
                if "#wolfram" in last_user_message:
                    tools.append("#wolfram")
                if "#image-gen" in last_user_message or "#image" in last_user_message:
                    tools.append("#image-gen")
                
                to_use_tools = []
                TOOL_PROMPT_VARIABLE_REPLACE = ""
                
                if len(tools) != 0:
                    for key, value in __tools__.items():
                        to_use_tools.append(
                            StructuredTool(
                                func=None,
                                name=key,
                                coroutine=value["callable"],
                                args_schema=value["pydantic_model"],
                                description=value["spec"]["description"],
                            )
                        )

                    def create_pydantic_model_from_docstring(func):
                        doc = inspect.getdoc(func)
                        if not doc:
                            return create_model(f"{func.__name__}Args")

                        param_descriptions = {}
                        for line in doc.split("\n"):
                            if ":param" in line:
                                param, desc = line.split(":param ", 1)[1].split(":", 1)
                                param = param.strip()
                                desc = desc.strip()
                                param_descriptions[param] = desc

                        type_hints = get_type_hints(func)
                        fields = {}
                        for param, hint in type_hints.items():
                            if param != "return":
                                fields[param] = (
                                    hint,
                                    Field(description=param_descriptions.get(param, "")),
                                )

                        return create_model(f"{func.__name__}Args", **fields)
                    
                    for tool in tools:
                        if tool == "#online":
                            TOOL_PROMPT_VARIABLE_REPLACE += WEB_SEARCH_PROMPT
                            online_tools = [
                                (self.search_web, "Search the internet for information."),
                                (self.scrape_website, "Get the contents of a website/url."),
                            ]
                            for func, desc in online_tools:
                                to_use_tools.append(
                                    StructuredTool(
                                        func=None,
                                        name=func.__name__,
                                        coroutine=func,
                                        args_schema=create_pydantic_model_from_docstring(
                                            func
                                        ),
                                        description=desc,
                                    )
                                )
                        if tool == "#python":
                            TOOL_PROMPT_VARIABLE_REPLACE += PYTHON_PROMPT
                            python_tools = [
                                (self.python_code_execution, "Execute Python code in a Jupyter kernel."),
                            ]
                            for func, desc in python_tools:
                                to_use_tools.append(
                                    StructuredTool(
                                        func=None,
                                        name=func.__name__,
                                        coroutine=func,
                                        args_schema=create_pydantic_model_from_docstring(
                                            func
                                        ),
                                        description=desc,
                                    )
                                )
                        if tool == "#image-gen":
                            TOOL_PROMPT_VARIABLE_REPLACE += IMAGE_GENERATION_PROMPT
                            image_gen_tools = [
                                (
                                    self.create_image_basic,
                                    "Generate an image based on a given prompt using the basic endpoint.",
                                ),
                                (
                                    self.create_image_pro,
                                    "Generate an image based on a given prompt using the pro endpoint.",
                                ),
                                (
                                    self.image_to_image,
                                    "Generate an image based on a given prompt using the image to image endpoint.",
                                ),
                            ]
                            for func, desc in image_gen_tools:
                                to_use_tools.append(
                                    StructuredTool(
                                        func=None,
                                        name=func.__name__,
                                        coroutine=func,
                                        args_schema=create_pydantic_model_from_docstring(
                                            func
                                        ),
                                        description=desc,
                                    )
                                )
                        if tool == "#wolfram":
                            TOOL_PROMPT_VARIABLE_REPLACE += WOLFRAM_ALPHA_PROMPT
                            wolfram_tools = [
                                (
                                    self.wolframAlpha,
                                    "Query the WolframAlpha knowledge engine to answer a wide variety of questions.",
                                )
                            ]
                            for func, desc in wolfram_tools:
                                to_use_tools.append(
                                    StructuredTool(
                                        func=None,
                                        name=func.__name__,
                                        coroutine=func,
                                        args_schema=create_pydantic_model_from_docstring(
                                            func
                                        ),
                                        description=desc,
                                    )
                                )
                
                if TOOL_PROMPT_VARIABLE_REPLACE == "": 
                    TOOL_PROMPT_VARIABLE_REPLACE = "No tools selected."
                     
                
                #
                # Setup Prompt and messages 
                # 
                
                input_messages = body["messages"]
                last_input_message = input_messages[-1]
                
                # Look for #small, #medium, #large, #reason or #online-model in the last message
                if "#small" in last_input_message["content"]:
                    model_from_answer = self.valves.AI_MODEL_LIST.split(";")[1]
                elif "#medium" in last_input_message["content"]:
                    model_from_answer = self.valves.AI_MODEL_LIST.split(";")[2]
                elif "#large" in last_input_message["content"]:
                    model_from_answer = self.valves.AI_MODEL_LIST.split(";")[3]
                elif "#reason" in last_input_message["content"]:
                    model_from_answer = self.valves.AI_MODEL_LIST.split(";")[4]
                elif "#online-model" in last_input_message["content"] or "#onlinemodel" in last_input_message["content"] or "#sonar" in last_input_message["content"]:
                    model_from_answer = self.valves.AI_MODEL_LIST.split(";")[5]
                    
                # Replace those words in the last message
                last_input_message["content"] = last_input_message["content"].replace("#small", "").replace("#medium", "").replace("#large", "").replace("#reason", "").replace("#online-model", "")
                
                # Remove first message from input messages such that only the messages after it remain
                input_messages.pop(0)
                # Remove last message from input messages such that only the messages before it remain
                input_messages.pop(-1)
                
                SYSTEM_PROMPT = REACT_SYSTEM_PROMPT
                
                # Tools
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{TOOLS}}", str(tools))
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{TOOL_USE_PROMPTS}}", TOOL_PROMPT_VARIABLE_REPLACE)
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{JUPYTER_URL}}", self.valves.JUPYTER_URL_BASE)
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{JUPYTER_TOKEN}}", self.valves.JUPYTER_TOKEN)
                
                # Date YYYY-MM-DD and time HH:MM:SS (UTC+x)
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{DATE}}", datetime.datetime.now().strftime("%Y-%m-%d"))
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{TIME}}", datetime.datetime.now().strftime("%H:%M:%S (%Z UTC%z)"))
                
                # User things
                SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{{USER_PROMT}}", self.valves.USER_PROMPT)
                     
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                for message in input_messages:
                    messages.append({"role": message["role"], "content": message["content"]})
                    
                messages.append({"role": last_input_message["role"], "content": last_input_message["content"]})
                
                # 
                # ReAct System
                #
                
                # Debug zields
                #yield f"tools = {tools}\n"
                #yield f"```json\n{json.dumps(messages, indent=2)}\n```\n\n"
                #yield f"```json\n{to_use_tools=}\n```\n\n"
                
                if model_from_answer != "perplexity/sonar-reasoning" and model_from_answer != "openai/3o-mini":
                    graph = create_react_agent(active_model, tools=to_use_tools)
                    inputs = {"messages": messages}

                    num_tool_calls = 0
                    async for event in graph.astream_events(
                        inputs, version="v2", config=config
                    ):
                        if num_tool_calls >= 6:
                            yield "[TOO MANY TOOL CALLS - AGENT TERMINATED]"
                            break
                        kind = event["event"]
                        data = event["data"]
                        if kind == "on_chat_model_stream":
                            if "chunk" in data and (content := data["chunk"].content):
                                yield content
                        elif kind == "on_tool_start":
                            yield "\n"
                            await send_status(f"Running tool {event['name']}", False)
                        elif kind == "on_tool_end":
                            num_tool_calls += 1
                            
                            # If tool is pyton print out python code in one code block and std_out and std_err in another
                            try: 
                                if event["name"] == "python_code_execution":
                                    # Check if output is a string that needs parsing
                                    output = str(data.get('output'))
                                    # Use regex to extract the JSON content
                                    pattern = r"content='(\{.*?\})'"
                                    match = re.search(pattern, output)
                                    
                                    output_json = match.group(1)
                                    output = json.loads(output_json)
                                    stdout = output.get("stdout", "")
                                    stderr = output.get("stderr", "")
                                    result = output.get("result", "")
                                    
                                    # Yield results to user
                                    yield f"\n```python\n{data['input']['code']}\n```\n"
                                    yield f"```bash\nSTDOUT: {stdout}\nSTDERR: {stderr}\nRESULT: {result}\n```\n"
                                    #yield f"```bash\n{output}\n```\n"
                            except: 
                                pass 
                                
                            await send_status(
                                f"Tool '{event['name']}' returned {data.get('output')}",
                                True,
                            )
                            await send_citation(
                                url=f"Tool call {num_tool_calls}",
                                title=event["name"],
                                content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                            )

                    await send_status(
                        status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_from_answer}." + 
                                    (f" Used {num_tool_calls} tools: {', '.join(tools)}." if num_tool_calls > 0 else ""),
                        done=True,
                    )
                elif model_from_answer == "perplexity/sonar-reasoning":
                    # Create a streaming OpenAI model with the Sonar reasoning model
                    client = OpenAI(
                        base_url=self.valves.OPENAI_BASE_URL,
                        api_key=self.valves.OPENAI_API_KEY,
                    )

                    completion = client.chat.completions.create(
                        model="perplexity/sonar-reasoning",
                        messages=messages,
                        stream=True,
                        tools=None,
                    )
                    
                    # Process the streaming response
                    citations = []
                    
                    # First yield the streaming content
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        
                        try: 
                            # Extract citation data if available
                            if hasattr(chunk, 'citations') and chunk.citations:
                                for citation in chunk.citations:
                                    # Citation is a string url, see if it is alreay in the list, otherwise add it
                                    if citation not in citations:
                                        citations.append(citation)
                        except Exception as e:
                            pass
                        
                    # After streaming is complete, yield the citations as formatted HTML
                    if citations:
                        yield "\n\n### Sources\n"
                        for i, citation in enumerate(citations):
                            formatted_citation = f"[{i+1}] [{citation}]({citation})"
                            yield formatted_citation + "\n"                        
                    await send_status(
                        status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_from_answer}.",
                        done=True,
                    )
                elif model_from_answer == "openai/3o-mini":
                    client = OpenAI(
                        base_url=self.valves.OPENAI_BASE_URL,
                        api_key=self.valves.OPENAI_API_KEY,
                    )

                    completion = client.chat.completions.create(
                        model="openai/3o-mini",
                        messages=messages,
                        stream=True,
                        tools=None,
                    )
                    
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                            
                    await send_status(
                        status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_from_answer}.",
                        done=True,
                    )
            
            except Exception as e:
                yield "Error: " + str(e)
                traceback.print_exc()
                return