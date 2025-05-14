# Search providers: Exa-py, Brave, Perplexity

from exa_py import Exa
import os
from openai import OpenAI

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

    return {"success": True, "data": result, "iframe": False}   

def crawl_exa(url: str):
    exa = Exa(api_key = os.getenv("EXA_API_KEY"))
    result = exa.get_contents([url], text = True)
    return {"success": True, "data": result, "iframe": False}   

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
    
    return {"success": True, "data": completion.choices[0].message.content, "iframe": False}
            
        
        