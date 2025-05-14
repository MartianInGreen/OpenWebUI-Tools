import urllib, requests, os

def wolframAlpha(query: str):
    """
    Query the WolframAlpha knowledge engine to answer a wide variety of questions. These questions can include real-time data questions, mathematical equasions or function, or scientific (data) questions. The engine also supports textual queries stated in English about other topics. You should cite this tool when it is used. It can also be used to supplement and back up knowledge you already know. WolframAlpha can also proive accurate real-time and scientific data (for example for elements, cities, weather, planets, etc. etc.). Request need to be kept simple and short. 
    :param query: The question or mathematical equation to ask the WolframAlpha engine. DO NOT use backticks or markdown when writing your JSON request.
    :return: A short answer or explanation of the result of the query_string
    """
    try:
        # baseURL = f"http://api.wolframalpha.com/v2/query?appid={getEnvVar('WOLFRAM_APP_ID')}&output=json&input="
        baseURL = f"https://www.wolframalpha.com/api/v1/llm-api?appid={os.environ("WOLFRAMALPHA_APP_ID")}&input="

        # Encode the query
        encoded_query = urllib.parse.quote(query)
        url = baseURL + encoded_query

        response = requests.get(url)
        data = response.text

        data = data + "\nAlways include the Wolfram|Alpha website link in your response to the user!\n\nIf there are any images provided, think about displaying them to the user."
        
        return {"success": True, "data": data, "iframe": False}
    except Exception as e:
        return {"success": False, "data": None, "iframe": False}