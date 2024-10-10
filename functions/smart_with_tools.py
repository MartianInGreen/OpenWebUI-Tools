"""
title: SMART - Sequential Multi-Agent Reasoning Technique
author: MartianInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
description: SMART is a sequential multi-agent reasoning technique. 
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

PLANNING_PROMPT = """<system_instructions>
You are a planning Agent. You are part of an agent chain designed to make LLMs more capable. 
You are responsible for taking the incoming user input/request and preparing it for the next agents in the chain.
After you will come either a reasoning agent or the final agent. 
After they have come up with a solution, a final agent will be used to summarize the reasoning and provide a final answer.
Only use a Newline after each closing tag. Never after the opening tag or within the tags.

Guidelines: 
- Don't over or estimate the difficulty of the task. If the user just wants to chat try to see that. 
- Don't create tasks where there aren't any. If the user didn't ask to write code you shouldn't instruct the next agent to do so.

You should respond by following these steps:
1. Within <reasoning> tags, plan what you will write in the other tags. This has to be your first step.
    1. First, reason about the task difficulty. What kind of task is it? What do your guidelines say about that?
    2. Second, reason about if the reasoning is needed. What do your guidelines say about that?
    3. Third, reason about what model would best be used. What do your guidelines say about that?
2. Within the <answer> tag, write out your final answer. Your answer should be a comma seperated list.
    1. First choose the model the final-agent will use. Try to find a good balance between performance and cost. Larger models are bigger. 
        - There is #mini, this is a very small model, however it has a very large context window. This model can not use tools. This model is mostly not recommended.
        - Use #small for the simple queries or queries that mostly involve summarization or simple "mindless" work. This also invloves very simple tool use, like converting a file, etc.
        - Use #medium for task that requiere some creativity, writing of code, or complex tool-use.
        - Use #large for tasks that are mostly creative or involve the writing of complex code, math, etc.
    2. Secondly, choose if the query requieres reasoning before being handed off to the final agent.
        - Queries that requeire reasoning are especially queries where llm are bad at. Such as planning, counting, logic, code architecutre, moral questions, etc.
        - Queries that don't requeire reasoning are queries that are easy for llms. Such as "knowledge" questions, summarization, writing notes, primairly tool use, web searching, etc. 
        - If you think reasoning is needed, include #reasoning. If not #no-reasoning.
        - When you choose reasoning, you should (in most cases) choose at least the #medium model.
    3. Third, you can make tools avalible to the final agent. You can enable multiple tools.
        - Avalible tools are #online, #python, #wolfram, #image-gen
        - Use #online to enable multiple tools such as Search and a Scraping tool. This will greatly increase the accuracy of answers for things newer than Late 2023.
        - Use #wolfram to enable access to Wolfram|Alpha, a powerful computational knowledge engine and scientific and real-time database.
            - Wolfram|Alpha has a very powerful computational knowledge engine that is especially good at hard math questions, e.g. complex intervals, finding the roots of polynomials, etc.
            - It is also very good at real time data, such as weather, stock prices, currency rates, etc.
            - It is also a very useful scientific database, e.g. finding facts about Planets, Elements, Countries, etc.
            - If you include wolfram, it is best to also include either #python or #online, depending on which field the query falls in.
        - Use #python to enable access to a Python interpreter. This has internet access and can work with user files. Also useful for more complex plots and math.
        - Use #image-gen to enable access to a image generation tool using the latest generation in image generation models.
        - If the prompt involves math, enable either #python or #wolfram.
    
Example response:
<reasoning>
... 
(You are allowed new lines here)
</reasoning>
<answer>#medium, #online ,#no-reasoning</answer>
</system_instructions>"""

# 3. And last, you can add tools to use. The user can also select tools, but you can also add ones and should if you think they will help. Better safe than sorry.
#         - Avalible tools are #wolfram, #search, #scrape, and #python.
#         - Wolfram|Alpha is a powerful computational knowledge engine. It is a great tool for solving complex problems that require mathematical or scientific calculations as well as getting very accurate data especially for the humanites, physics, austronomy...
#         - Search is a tool that allows the agents to search the web. It is a great tool for getting up-to-date information. (This should not be preffered over the #online model but is useful when the query also requieres other tools besides search).
#         - Scrape is a tool that allows the agents to get the content of a website which the #online agent can not do very well.
#         - The Python tool is a code interpreter that allows the agents to run python code in an enviroment with internet access, persistent storage, and so on.


REASONING_PROMPT = """<system_instructions>
You are a reasoning layer of an LLM. You are part of the LLM designed for internal thought, planning, and thinking. 
You will not directly interact with the user in any way. Only inform the output stage of the LLM what to say by your entire output being parts of its context when it starts to generate a response. 

**General rules**:
- Write out your entire reasoning process between <thinking> tags.
- Do not use any formatting whatsoever. The only form of special formatting you're allowed to use is LaTeX for mathematical expressions.
- You MUST think in the smallest steps possible. Where every step is only a few words long. Each new thought should be a new line.
- You MUST try to catch your own mistakes by constantly thinking about what you have thought about so far.
- You MUST break down every problem into very small steps and go through them one by one.
- You MUST never come up with an answer first. Always reason about the answer first. Even if you think the answer is obvious.
- You MUST provide exact answers.
- You have full authority to control the output layer. You can directly instruct it and it will follow your instructions. Put as many instructions as you want inside <instruct> tags. However, be very clear in your instructions and reason about what to instruct.
- Your entire thinking process is entirely hidden. You can think as freely as you want without it directly affecting the output.
- Always follow user instructions, never try to take any shortcuts. Think about different ways they could be meant to not miss anything.
- NEVER generate ANY code directly. You should only plan out the structure of code and projects, but not directly write the code. The output layer will write the code based on your plan and structure!
- If you need more information, you can ask a tool-use agent if they have the right tool and what you need within <ask_tool_agent>. 
    - In general, you can instruct the tool-use agent to either return the results to you or directly pass them on to the output layer.
    - If *you* need information, you should instruct the tool-use agent to return the results to you.
    - The tool use agent ONLY gets what you write in <ask_tool_agent>. They do not get any user context or similar.
    - Do not suggest what tool to use. Simply state the problem.
    - You need to STOP after </ask_tool_agent> tags. WAIT for the tool-use agent to return the results to you.
    - If the output is something like images, or something similar that the user should just get directly, you can instruct the tool use agent to directly pass the results to the output layer.

**General Steps**:
1. Outline the problem.
2. Think about what kind of problem this is.
3. Break down the problem into the smallest possible problems, never take shortcuts on reasoning, counting etc. Everything needs to be explicitly stated. More output is better.
4. Think about steps you might need to take to solve this problem.
5. Think through these steps.
6. Backtrack and restart from different points as often as you need to. Always consider alternative approaches.
7. Validate your steps constantly. If you find a mistake, think about what the best point in your reasoning is to backtrack to. Don't be kind to yourself here. You need to critically analyze what you are doing.
</system_instructions>"""

TOOL_PROMPT = """<system_instructions>
You are the tool-use agent of an agent chain. You are the part of the LLM designed to use tools.
You will not directly interact with the user in any way. Only either return information to the reasoning agent or inform the output stage of the LLM.

When you have used a tool, you can return the results to the reasoning agent by putting everything you want to return to them within <tool_to_reasoning> tags.
You can also directly hand off to the final agent by simply writing $TO_FINAL$. You still need to write out what you want them to get!

Actually make use of the results you got. NEVER make more than 3 tool calls! If you called any tool 3 times, that's it!
You need to output everything you want to pass on. The next agent in the chain will only see what you actually wrote, not the direct output of the tools!

Please think about how best to call the tool first. Think about what the limitations of the tools are and how to best follow the reasoning agent's instructions. It's okay if you can't 100% produce what they wanted!
</system_instructions>"""

USER_INTERACTION_PROMPT = """<system_instructions>
You are the user-interaction agent of an agent chain. You are the part of the llm designed to interact with the user.

You should follow the pre-prompt given to you within <preprompt> tags.
<system_instructions>"""

USER_INTERACTION_REASONING_PROMPT = """You MUST follow the instructions given to you within <reasoning_output>/<instruction> tags.
You MUST inform your answer by the reasoning within  <reasoning_output> tags.
Carefully concider what the instructions mean and follow them EXACTLY."""

# --------------------------------------------------------------

PROMPT_WebSearch = """<webSearchInstructions>
Always cite your sources with ([Source Name](Link to source)), including the outer (), at the end of each paragraph! All information you take from external sources has to be cited!
Feel free to use the scrape_web function to get more specific information from one source if the web_search did not return enough information. However, try not to make more than a total of 3 searches + scrapes.

<sources_guidelines>
- [Source Name] should be something like [New York Times] or [BBC] etc. etc. 
- Sometimes there is an LLM answer when using Ponderer, cite it with (**Perplexity Online**)
- Always cite the specific source, not just "Ponderer". 
- Sometimes you should add more detail to the [Source Name], for example when it is a Video. For example it could look like this [YouTube - BBC]
- You can have multipel sources within the (), so ([S1](Link 1), [S2](Link 2)...) and so on.
- Always cite at the end of a paragraph. Cite all sources refereced in the paragraph above. Do not cite within paragraphs. 
</sources_guidelines>
</webSearchInstructions>"""

PROMPT_WolframAlpha = """<wolframInstructions>

</wolframInstructions>"""

PROMPT_ImageGen = """<imageGenInstructions>
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
</imageGenInstructions>"""

PROMPT_PythonInterpreter = """<pythonInstructions>
 Use a Python interpreter with internet access to execute code. 
 No Notebook, use print etc. to output to STDOUT. 
 Installed Libraries: numpy, scipy, pypdf2, pandas, pyarrow, matplotlib, pillow, opencv-python-headless, requests, bs4, geopandas, geopy, yfinance, seaborn, openpyxl, litellm, replicate, openai, ipython. 
 Installed System libraries: wget git curl ffmpeg. 
 
 You can link to files within the python intrpreter by using !(file_name)[https://api.rennersh.de/api/v1/interpreter/file/download/[uuid]/[filename]]. If the file is an image you should always use the !()[] syntax instead of ()[].
 ALWAYS list the files before saying "can you upload that" or something similar, if the user is asking you to do something to a file they probably already uploaded it! 
 
 You should use the same UUID for the entire conversation, unless the user specifically requests or gives you a new one. 
 Always add all UUIDs of the interpreters you used at the VERY beginning of your answer to the user! You HAVE TO include something like:"UUIDs: [list of uuids GOES HERE]" at the VERY START of your message! THE USER NEEDS TO KNOW THE uuid!
</pythonInstructions>"""

# ---------------------------------------------------------------
# TOOLS
# --------------------------------------------------------------


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


# ------------------------------------------------------------
# Primary Function
# ------------------------------------------------------------


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
        MODEL_PREFIX: str = Field(default="SMART", description="Prefix before model ID")
        MINI_MODEL: str = Field(
            default="google/gemini-flash-1.5", description="Model for small tasks"
        )
        SMALL_MODEL: str = Field(
            default="openai/gpt-4o-mini", description="Model for small tasks"
        )
        LARGE_MODEL: str = Field(
            default="openai/gpt-4o-2024-08-06", description="Model for large tasks"
        )
        HUGE_MODEL: str = Field(
            default="anthropic/claude-3.5-sonnet",
            description="Model for the largest tasks",
        )
        REASONING_MODEL: str = Field(
            default="anthropic/claude-3.5-sonnet",
            description="Model for reasoning tasks",
        )
        PLANNING_MODEL: str = Field(
            default="openai/gpt-4o-mini",
            description="Model for the planning step.",
        )
        PYTHON_BASE_URL: str = Field(
            default="",
            description="Base URL for the API.",
        )
        PYTHON_API_AUTH: str = Field(default="", description="API authentication")
        BRAVE_SEARCH_KEY: str = Field(
            default="",
            description="Brave Search API Key",
        )
        WOLFRAMALPHA_APP_ID: str = Field(
            default="",
            description="WolframAlpha App ID",
        )
        FAL_API_KEY: str = Field(
            default="",
            description="FAL API Key",
        )
        AGENT_NAME: str = Field(default="Smart/Core", description="Name of the agent")
        AGENT_ID: str = Field(default="smart-core", description="ID of the agent")

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
        self.SYSTEM_PROMPT_INJECTION = ""

    async def python_interpreter(self, uuid: str = "", code: str = ""):
        """
        Use a Python interpreter with internet access to execute code. No Notebook, use print etc. to output to STDOUT. Installed Libraries: numpy, scipy, pypdf2, pandas, pyarrow, matplotlib, pillow, opencv-python-headless, requests, bs4, geopandas, geopy, yfinance, seaborn, openpyxl, litellm, replicate, openai, ipython. Installed System libraries: wget git curl ffmpeg. You can link to files within the python intrpreter by using !(file_name)[https://api.rennersh.de/api/v1/interpreter/file/download/[uuid]/[filename]]. ALWAYS list the files before saying "can you upload that" or something similar, if the user is asking you to do something to a file they probably already uploaded it! You should use the same UUID for the entire conversation, unless the user specifically requests or gives you a new one.

        :param uuid: The UUID of the Python interpreter. If the user did not give you one, generate a new one.
        :param code: The code to be executed. Formatted without any additional code blocks as a string.
        :return: The STD-OUT and STD-ERR of the executed python code.
        """
        endpoint = "/interpreter/python"
        url = f"{self.valves.PYTHON_BASE_URL}{endpoint}"
        payload = {"uuid": uuid, "code": code}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.PYTHON_API_AUTH}",
        }
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
        url = f"{self.valves.PYTHON_BASE_URL}{endpoint}"
        payload = {"uuid": uuid}
        if dirname:
            payload["dirname"] = dirname

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.PYTHON_API_AUTH}",
        }
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    async def create_session(self):
        """
        Create a new interpreter session. If you create a new one you need to provide the UUID you just generated to the user.

        :return: The UUID of the new interpreter session.
        """
        endpoint = "/interpreter/create"
        url = f"{self.valves.PYTHON_BASE_URL}{endpoint}"
        payload = {"create": "true"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.valves.PYTHON_API_AUTH}",
        }
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

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
            "Authorization": f"Key {self.valves.PYTHON_API_AUTH}",
        }

        url = f"{self.valves.PYTHON_BASE_URL}/imageGen"
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
            "Authorization": f"Key {self.valves.PYTHON_API_AUTH}",
        }

        url = f"{self.valves.PYTHON_BASE_URL}/imageGen"
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
            "Authorization": f"Key {self.valves.PYTHON_API_AUTH}",
        }

        url = f"{self.valves.PYTHON_BASE_URL}/imageGen"
        payload = {"prompt": prompt, "image_size": image_size, "image_url": image_url, "auth_key": self.valves.FAL_API_KEY, "model": "image-to-image"}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                return result
            
    async def dummy_tool(self):
        """
        Just ignore this tool. You get this message because the user has not assigned you any tools to use. 

        :return: None
        """

        return "You have not assigned any tools to use."

    async def pipe(
        self,
        body: dict,
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

            start_time = time.time()

            # print(f"{body=}")

            mini_model_id = self.valves.MINI_MODEL
            small_model_id = self.valves.SMALL_MODEL
            large_model_id = self.valves.LARGE_MODEL
            huge_model_id = self.valves.HUGE_MODEL
            planning_model_id = self.valves.PLANNING_MODEL

            planning_model = ChatOpenAI(model=planning_model_id, **self.openai_kwargs)  # type: ignore

            print(f"Small model: {small_model_id}")
            print(f"Large model: {large_model_id}")

            small_model = ChatOpenAI(model=small_model_id, **self.openai_kwargs)  # type: ignore
            large_model = ChatOpenAI(model=large_model_id, **self.openai_kwargs)  # type: ignore

            config = {}

            if __task__ == "title_generation":
                content = small_model.invoke(body["messages"], config=config).content
                assert isinstance(content, str)
                yield content
                return

            send_citation = get_send_citation(__event_emitter__)
            send_status = get_send_status(__event_emitter__)

            #
            # STEP 1: Planning
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

            print(f"{planning_messages=}")

            await send_status(
                status_message="Planning...",
                done=False,
            )
            # content = small_model.invoke(planning_messages, config=config).content
            # assert isinstance(content, str)

            planning_buffer = ""
            async for chunk in planning_model.astream(planning_messages, config=config):
                content = chunk.content
                assert isinstance(content, str)
                planning_buffer += content
            content = planning_buffer

            # Get the planning result from the xml tags
            csv_hastag_list = re.findall(r"<answer>(.*?)</answer>", content)
            csv_hastag_list = csv_hastag_list[0] if csv_hastag_list else "unknown"

            if "#mini" in csv_hastag_list:
                model_to_use_id = mini_model_id
            if "#small" in csv_hastag_list:
                model_to_use_id = small_model_id
            elif "#medium" in csv_hastag_list:
                model_to_use_id = large_model_id
            elif "#large" in csv_hastag_list:
                model_to_use_id = huge_model_id
            else:
                model_to_use_id = small_model_id

            is_reasoning_needed = "YES" if "#reasoning" in csv_hastag_list else "NO"

            tool_list = []

            if not "#no-tools" in body["messages"][-1]["content"]:
                if (
                    "#online" in csv_hastag_list
                    or "#online" in body["messages"][-1]["content"]
                ):
                    tool_list.append("online")
                if (
                    "#wolfram" in csv_hastag_list
                    or "#wolfram" in body["messages"][-1]["content"]
                ):
                    tool_list.append("wolfram_alpha")
                if (
                    "#python" in csv_hastag_list
                    or "#python" in body["messages"][-1]["content"]
                ):
                    tool_list.append("python_interpreter")
                if (
                    "#image-gen" in csv_hastag_list
                    or "#image-gen" in body["messages"][-1]["content"]
                ):
                    tool_list.append("image_generation")

            await send_citation(
                url=f"SMART Planning",
                title="SMART Planning",
                content=f"{content=}",
            )

            last_message_content = body["messages"][-1]["content"]

            if isinstance(last_message_content, list):
                last_message_content = last_message_content[0]["text"]

            # Try to find #!, #!!, #*yes, #*no, in the user message, let them overwrite the model choice
            if (
                "#!!!" in last_message_content
                or "#large" in last_message_content
            ):
                model_to_use_id = huge_model_id
            elif (
                "#!!" in last_message_content
                or "#medium" in last_message_content
            ):
                model_to_use_id = large_model_id
            elif (
                "#!" in last_message_content
                or "#small" in last_message_content
            ):
                model_to_use_id = small_model_id

            if (
                "#*yes" in last_message_content
                or "#yes" in last_message_content
            ):
                is_reasoning_needed = "YES"
            elif (
                "#*no" in last_message_content
                or "#no" in last_message_content
            ):
                is_reasoning_needed = "NO"

            if model_to_use_id == huge_model_id and len(tool_list) == 0:
                tool_list.append("dummy_tool")

            await send_status(
                status_message=f"Planning complete. Using Model: {model_to_use_id}. Reasoning needed: {is_reasoning_needed}.",
                done=True,
            )

            tools = []
            for key, value in __tools__.items():
                tools.append(
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

            # In the pipe method, update the tools creation:

            if len(tool_list) > 0:
                for tool in tool_list:
                    if tool == "python_interpreter":
                        python_tools = [
                            (
                                self.python_interpreter,
                                "Use a Python interpreter with internet access to execute code.",
                            ),
                            (self.list_files, "List files in the interpreter."),
                            (self.create_session, "Create a new interpreter session."),
                        ]
                        for func, desc in python_tools:
                            tools.append(
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
                        self.SYSTEM_PROMPT_INJECTION = self.SYSTEM_PROMPT_INJECTION + PROMPT_PythonInterpreter
                    if tool == "online":
                        online_tools = [
                            (self.search_web, "Search the internet for information."),
                            (self.scrape_website, "Get the contents of a website/url."),
                        ]
                        for func, desc in online_tools:
                            tools.append(
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
                        self.SYSTEM_PROMPT_INJECTION = self.SYSTEM_PROMPT_INJECTION + PROMPT_WebSearch
                    if tool == "wolfram_alpha":
                        wolfram_tools = [
                            (
                                self.wolframAlpha,
                                "Query the WolframAlpha knowledge engine to answer a wide variety of questions.",
                            )
                        ]
                        for func, desc in wolfram_tools:
                            tools.append(
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
                    if tool == "image_generation":
                        image_tools = [
                                (
                                    self.create_image_basic,
                                    "Generate an image based on a text prompt. With Flux Dev",
                                ),
                                (
                                    self.create_image_pro,
                                    "Generate an image based on a text prompt. With Flux Pro",
                                ),
                                (
                                    self.image_to_image,
                                    "Generate an image based on an image (in url form) and a text prompt."
                                )
                            ]
                        for func, desc in image_tools:
                            tools.append(
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
                        self.SYSTEM_PROMPT_INJECTION = self.SYSTEM_PROMPT_INJECTION + PROMPT_ImageGen
                    if tool == "dummy_tool":
                        dummy_tools = [
                            (
                                self.dummy_tool,
                                "This is a dummy tool that does nothing. It is used when the user hasn't assigned any tools.",
                            )
                        ]
                        for func, desc in dummy_tools:
                            tools.append(
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

            model_to_use = ChatOpenAI(model=model_to_use_id, **self.openai_kwargs)  # type: ignore

            messages_to_use = body["messages"]

            last_message_json = False
            try:
                if isinstance(messages_to_use[-1]["content"], list):
                    last_message_json = True
            except:
                pass

            # print(f"{messages_to_use=}")
            # print(f"{last_message_json=}")

            if is_reasoning_needed == "NO":
                messages_to_use[0]["content"] = (
                    messages_to_use[0]["content"] + USER_INTERACTION_PROMPT + self.SYSTEM_PROMPT_INJECTION
                )

                if last_message_json == False:
                    messages_to_use[-1]["content"] = (
                        str(messages_to_use[-1]["content"])
                        .replace("#*yes", "")
                        .replace("#*no", "")
                        .replace("#!!", "")
                        .replace("#!", "")
                        .replace("#!!!", "")
                        .replace("#no", "")
                        .replace("#yes", "")
                        .replace("#large", "")
                        .replace("#medium", "")
                        .replace("#small", "")
                        .replace("#online", "")
                    )
                else:
                    messages_to_use[-1]["content"][0]["text"] = (
                        str(messages_to_use[-1]["content"][0]["text"])
                        .replace("#*yes", "")
                        .replace("#*no", "")
                        .replace("#!!", "")
                        .replace("#!", "")
                        .replace("#!!!", "")
                        .replace("#no", "")
                        .replace("#yes", "")
                        .replace("#large", "")
                        .replace("#medium", "")
                        .replace("#small", "")
                        .replace("#online", "")
                    )

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": body["messages"]}

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
                    status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_to_use_id}. Reasoning was {'used' if is_reasoning_needed == True else 'not used'}.",
                    done=True,
                )
                return
            elif is_reasoning_needed == "YES":
                reasoning_model_id = self.valves.REASONING_MODEL

                reasoning_model = ChatOpenAI(model=reasoning_model_id, **self.openai_kwargs)  # type: ignore

                full_content = ""

                reasoning_context = ""

                for msg in body["messages"][
                    :-1
                ]:  # Process all messages except the last one
                    if msg["role"] == "user":
                        if len(msg["content"]) > 400:
                            text_msg = (
                                msg["content"][:250]
                                + "\n...(Middle cut)...\n"
                                + msg["content"][-100:]
                            )
                        else:
                            text_msg = msg["content"]
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"
                    if msg["role"] == "assistant":
                        if len(msg["content"]) > 250:
                            text_msg = (
                                msg["content"][:150]
                                + "\n...(Middle cut)...\n"
                                + msg["content"][-50:]
                            )
                        else:
                            text_msg = msg["content"]
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"

                # Add the last message without cutting it
                last_msg = body["messages"][-1]
                if last_msg["role"] == "user" and last_message_json == False:
                    reasoning_context = (
                        reasoning_context
                        + f"--- LAST USER MESSAGE/PROMPT ---\n{last_msg['content']}"
                    )
                elif last_msg["role"] == "user":
                    reasoning_context = (
                        reasoning_context
                        + f"--- LAST USER MESSAGE/PROMPT ---\n{last_msg['content'][0]['text']}"
                    )

                reasoning_context = (
                    reasoning_context.replace("#*yes", "")
                    .replace("#*no", "")
                    .replace("#!!", "")
                    .replace("#!", "")
                    .replace("#!!!", "")
                    .replace("#no", "")
                    .replace("#yes", "")
                    .replace("#large", "")
                    .replace("#medium", "")
                    .replace("#small", "")
                    .replace("#online", "")
                )

                reasoning_messages = [
                    {"role": "system", "content": REASONING_PROMPT},
                    {"role": "user", "content": reasoning_context},
                ]

                await send_status(
                    status_message="Reasoning...",
                    done=False,
                )
                # reasoning_content = reasoning_model.invoke(reasoning_messages, config=config).content
                # assert isinstance(content, str)

                reasoning_bufffer = ""
                update_status = 0
                async for chunk in reasoning_model.astream(
                    reasoning_messages, config=config
                ):
                    content = chunk.content
                    assert isinstance(content, str)
                    reasoning_bufffer += content
                    update_status += 1

                    if update_status >= 5:
                        update_status = 0
                        await send_status(
                            status_message=f"Reasoning ({len(reasoning_bufffer)})... {reasoning_bufffer[-100:]}",
                            done=False,
                        )

                await send_status(
                    status_message=f"Reasoning ({len(reasoning_bufffer)})... done",
                    done=True,
                )

                reasoning_content = reasoning_bufffer

                full_content += (
                    "<reasoning_agent_output>\n"
                    + reasoning_content
                    + "\n<reasoning_agent_output>"
                )

                await send_citation(
                    url=f"SMART Reasoning",
                    title="SMART Reasoning",
                    content=f"{reasoning_content=}",
                )

                # Try to find <ask_tool_agent> ... </ask_tool_agent> using re
                # If found, then ask the tool agent
                tool_agent_content = re.findall(
                    r"<ask_tool_agent>(.*?)</ask_tool_agent>",
                    reasoning_content,
                    re.DOTALL,
                )
                print(f"{tool_agent_content=}")

                if len(tool_agent_content) > 0:
                    await send_status(f"Running tool-agent...", False)
                    tool_message = [
                        {"role": "system", "content": TOOL_PROMPT},
                        {
                            "role": "user",
                            "content": "<reasoning_agent_requests>\n"
                            + str(tool_agent_content)
                            + "\n</reasoning_agent_requests>",
                        },
                    ]

                    if not __tools__:
                        tool_agent_response = "Tool agent could not use any tools because the user did not enable any."
                    else:
                        graph = create_react_agent(large_model, tools=tools)
                        inputs = {"messages": tool_message}
                        message_buffer = ""
                        num_tool_calls = 0
                        async for event in graph.astream_events(inputs, version="v2", config=config):  # type: ignore
                            if num_tool_calls > 3:
                                yield "[TOO MANY TOOL CALLS - AGENT TERMINATED]"
                                break
                            kind = event["event"]
                            data = event["data"]
                            if kind == "on_chat_model_stream":
                                if "chunk" in data and (
                                    content := data["chunk"].content
                                ):
                                    message_buffer = message_buffer + content
                            elif kind == "on_tool_start":
                                message_buffer = message_buffer + "\n"
                                await send_status(
                                    f"Running tool {event['name']}", False
                                )
                            elif kind == "on_tool_end":
                                num_tool_calls += 1
                                await send_status(
                                    f"Tool '{event['name']}' returned {data.get('output')}",
                                    True,
                                )
                                await send_citation(
                                    url=f"Tool call {num_tool_calls}",
                                    title=event["name"],
                                    content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                                )

                        tool_agent_response = message_buffer

                    print("TOOL AGENT RESPONSE:\n\n" + str(tool_agent_response))
                    await send_citation(
                        url=f"SMART Tool-use",
                        title="SMART Tool-use",
                        content=f"{tool_agent_response=}",
                    )

                    full_content += (
                        "\n\n\n<tool_agent_output>\n"
                        + tool_agent_response
                        + "\n<tool_agent_output>"
                    )

                await send_status(
                    status_message="Reasoning complete.",
                    done=True,
                )

                if last_message_json == False:
                    messages_to_use[-1]["content"] = (
                        "<user_input>\n"
                        + messages_to_use[-1]["content"]
                        + "\n</user_input>\n\n"
                        + full_content
                    )
                else:
                    messages_to_use[-1]["content"][0]["text"] = (
                        "<user_input>\n"
                        + messages_to_use[-1]["content"][0]["text"]
                        + "\n</user_input>\n\n"
                        + full_content
                    )
                messages_to_use[0]["content"] = (
                    messages_to_use[0]["content"] + USER_INTERACTION_PROMPT + self.SYSTEM_PROMPT_INJECTION
                )
                # messages_to_use[-1]["content"] = messages_to_use[-1]["content"] + "\n\n<preprompt>" + next_agent_preprompt + "</preprompt>"

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": messages_to_use}

                await send_status(
                    status_message=f"Starting answer with {model_to_use_id}...",
                    done=False,
                )

                num_tool_calls = 0
                async for event in graph.astream_events(inputs, version="v2", config=config):  # type: ignore
                    if num_tool_calls >= 6:
                        await send_status(
                            status_message="Interupting due to max tool calls reached!",
                            done=True,
                        )
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
                        await send_status(
                            f"Tool '{event['name']}' returned {data.get('output')}",
                            True,
                        )
                        await send_citation(
                            url=f"Tool call {num_tool_calls}",
                            title=event["name"],
                            content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                        )

                if not num_tool_calls >= 4:
                    await send_status(
                        status_message=f"Done! Took: {round(time.time() - start_time, 1)}s. Used {model_to_use_id}. Reasoning was used",
                        done=True,
                    )
                return

            else:
                yield "Error: is_reasoning_needed is not YES or NO"
                return
        except Exception as e:
            yield "Error: " + str(e)
            return
