"""
title: Web Search
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
"""

import urllib, requests, os, json, time, re
from dotenv import load_dotenv #type: ignore

# Prettier #print
from rich import print
from pydantic import BaseModel, Field #type: ignore
from typing import Callable, Awaitable


class Tools:
    class Valves(BaseModel):
        BRAVE_SEARCH_KEY: str = Field(
            default="",
            description="Brave Search API Key.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True
        os.environ["BRAVE_SEARCH_TOKEN"] = self.valves.BRAVE_SEARCH_KEY
        os.environ["BRAVE_SEARCH_TOKEN_SECONDARY"] = self.valves.BRAVE_SEARCH_KEY

    # ------------------------------------------------------------


# Helper functions
# ------------------------------------------------------------


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


class Tools:
    class Valves(BaseModel):
        SEARCH_KEY: str = Field(
            default="",
            description="Brave Search API Key",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

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

        results = searchWeb(query, country, language, focus, self.valves.SEARCH_KEY)
        print(results)
        return json.dumps(results)
