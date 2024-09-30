"""
title: WolframAlpha API
author: MartainInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.1.0
requirements: pytube, youtube_transcript_api
"""

import json
from pytubefix import YouTube #type: ignore
from youtube_transcript_api import YouTubeTranscriptApi #type: ignore
from pydantic import BaseModel, Field #type: ignore
from typing import Callable, Awaitable

def youtube_func(video_id: str):
    #print(f"Getting video with id: {video_id}")
    yt = YouTube(f'https://www.youtube.com/watch?v={video_id}', use_po_token=True)

    try:
        # Get avalilbe languages
        languages_raw = YouTubeTranscriptApi.list_transcripts(video_id)
        languages = []

        for lang in languages_raw:
            languages.append(lang.language_code)
        #print(f"Available languages: {languages}")

        # Get transcript (get english by default, if not available get first language)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[languages[0]])
   
        text = []
        for part in transcript:
            text.append(part['text'])

        print(text)
    except: 
        text = "Could not get video transcript :("

    try:
        i: int = yt.watch_html.find('"shortDescription":"')
        desc: str = '"'
        i += 20  # excluding the `"shortDescription":"`
        while True:
            letter = yt.watch_html[i]
            desc += letter  # letter can be added in any case
            i += 1
            if letter == '\\':
                desc += yt.watch_html[i]
                i += 1
            elif letter == '"':
                break
        
        return {
            'title': yt.title,
            'channel': yt.author,
            'description': desc,
            'length': str(yt.length) + " s",
            'views': yt.views,
            'transcription': text
        }
    except Exception as e:
        print(str(e))
        return {"Something went wrong :("}
    
class Tools:
    class Valves(BaseModel):
        WOLFRAMALPHA_APP_ID: str = Field(
            default="",
            description="The App ID (api key) to authorize WolframAlpha",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    def youtube(
        self, video_id: str, __event_emitter__: Callable[[dict], Awaitable[None]]
    ) -> str:
        """
        This function lets you get information about YouTube videos. Including Metadata and Transcription.
        :param video_id: Video ID of the YouTube video
        :return: A short answer or explanation of the result of the query_string
        """

        data = youtube_func(video_id)

        return json.dumps(data)