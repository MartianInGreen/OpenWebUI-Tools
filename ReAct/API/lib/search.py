# Search providers: Exa-py, Brave, Perplexity

from exa_py import Exa
import os
from openai import OpenAI
import pathlib
import json
import urllib.parse

def get_iframe_content(data):
    iframe_path = pathlib.Path(__file__).parent / "iframes" / "search.html"
    with open(iframe_path, 'r') as f:
        content = f.read()
        
    # Convert data to JSON string and escape it for JavaScript
    data_json = json.dumps(data).replace('</', '<\\/')
    
    # Insert the data into the HTML content
    content = content.replace('<!-- Results will be populated here -->', 
                            f'<script>displayResults({data_json});</script>')
    return content

def create_iframe_url(data):
    # Encode the data as a URL parameter
    encoded_data = urllib.parse.quote(json.dumps(data))
    return f"search.html?data={encoded_data}"

def serach_exa(query: str, results_category: str, publish_date: list[str]):
    exa = Exa(api_key = os.getenv("EXA_API_KEY"))
    
    params = {
        "text": True
    }
    
    if results_category:
        params["results_category"] = results_category
        
    if publish_date:
        params["start_published_date"] = publish_date[0]
        params["end_published_date"] = publish_date[1]
        
        
    result = exa.search_and_contents(query, **params)
    data = {"success": True, "data": result, "iframe": True}
    
    return {
        "success": True, 
        "data": result, 
        "iframe": True,
        "iframe_content": get_iframe_content(data)
    }   

def crawl_exa(url: str):
    exa = Exa(api_key = os.getenv("EXA_API_KEY"))
    result = exa.get_contents([url], text = True)
    data = {"success": True, "data": result, "iframe": True}
    
    return {
        "success": True, 
        "data": result, 
        "iframe": True,
        "iframe_content": get_iframe_content(data)
    }   

def serach_brave(query):
    pass 

def serach_perplexity(query: str, type: str):
    # Search Perplexity using the Openrouter api
    types = ["fast", "normal", "deep"]
    if type not in types:
        raise ValueError(f"Invalid type: {type}")
    
    # Match type to model 
    if type == "fast":
        model = "perplexity/sonar"
    elif type == "normal":
        model = "perplexity/sonar-reasoning"
    elif type == "deep":
        model = "perplexity/sonar-deep-research"
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    
    
    completion = client.chat.completions.create(
        extra_headers={},
        extra_body={},
        model=model,
        messages=[
            {
            "role": "user",
            "content": query
            }
        ]
    )
    
    return {
        "success": True, 
        "data": completion.choices[0].message.content, 
        "iframe": False
    }
            
        
        