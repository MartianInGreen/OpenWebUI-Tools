"""
title: YouTube Utility Tools
author: MartainInGreen, Firkin-gadabout
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
version: 0.2.0
requirements: youtube_transcript_api, google-api-python-client, pydantic

This module provides tools for interacting with YouTube via Data API v3
and fetching transcripts via the youtube_transcript_api.

Available tools:
  - Tools.transcript_download(video_id)  : Download video metadata and full transcript.
  - Tools.search(query, max_results)    : Search YouTube for videos matching a keyword.

All tools return only JSON-serializable types and support optional streaming
via an `__event_emitter__` callback for incremental output. Search results
are cached with LRU (maxsize=128) and enforce max_results bounds.
"""

import json
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
from pydantic import BaseModel, Field  # type: ignore
from typing import Callable, Awaitable, List, Dict
from googleapiclient.discovery import build  # type: ignore
from functools import lru_cache


class TranscriptDownloadResult(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel or author name")
    description: str = Field(..., description="Full video description")
    duration: str = Field(..., description="ISO8601 duration (e.g. 'PT5M33S')")
    view_count: str = Field(..., description="Total view count as string")
    transcription: List[str] = Field(
        ..., description="List of transcript text segments"
    )


class SearchItem(BaseModel):
    video_id: str = Field(..., description="Unique YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel or uploader name")
    published_at: str = Field(..., description="ISO8601 publication timestamp")
    description: str = Field(..., description="Snippet description")
    view_count: str = Field(..., description="Total view count as string")
    like_count: str = Field(..., description="Total like count as string")
    comment_count: str = Field(..., description="Total comment count as string")
    length: str = Field(..., description="ISO8601 duration of the video")


class SearchResult(BaseModel):
    results: List[SearchItem] = Field(..., description="List of search results")


class Tools:
    class Valves(BaseModel):
        YOUTUBE_API_KEY: str = Field(
            "", description="YouTube Data API v3 key for all API calls"
        )

    def __init__(self):
        """
        Initialize the Tools container.

        - Set `valves.YOUTUBE_API_KEY` before calling any methods.
        - `self.citation = True` indicates support for citing results.
        """
        self.valves = self.Valves()
        self.citation = True

    def transcript_download(
        self,
        video_id: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Dict:
        """
        Download metadata and full transcript for a given YouTube video.

        Returns a dict matching TranscriptDownloadResult schema.
        """
        # 1) Fetch transcript
        try:
            transcripts = YouTubeTranscriptApi.get_transcript(video_id)
            transcription = [seg.get("text", "") for seg in transcripts]
        except Exception:
            transcription = []

        # 2) Fetch metadata
        youtube = build("youtube", "v3", developerKey=self.valves.YOUTUBE_API_KEY)
        try:
            resp = (
                youtube.videos()
                .list(part="snippet,contentDetails,statistics", id=video_id)
                .execute()
            )
            items = resp.get("items", [])
            if items:
                item = items[0]
                sn = item.get("snippet", {})
                cd = item.get("contentDetails", {})
                st = item.get("statistics", {})
                data = {
                    "video_id": video_id,
                    "title": sn.get("title", ""),
                    "channel": sn.get("channelTitle", ""),
                    "description": sn.get("description", ""),
                    "duration": cd.get("duration", ""),
                    "view_count": st.get("viewCount", "0"),
                    "transcription": transcription,
                }
            else:
                data = {
                    "video_id": video_id,
                    "title": "",
                    "channel": "",
                    "description": "",
                    "duration": "",
                    "view_count": "0",
                    "transcription": transcription,
                }
        except Exception:
            data = {
                "video_id": video_id,
                "title": "",
                "channel": "",
                "description": "",
                "duration": "",
                "view_count": "0",
                "transcription": transcription,
            }

        # Validate & serialize via Pydantic
        result = TranscriptDownloadResult(**data).dict()
        if __event_emitter__:
            __event_emitter__(
                {"type": "message", "data": {"content": json.dumps(result)}}
            )
        return result

    @lru_cache(maxsize=128)
    def _search_logic(self, query: str, max_results: int) -> List[Dict]:
        # Validate bounds
        if not 1 <= max_results <= 50:
            raise ValueError("max_results must be between 1 and 50")
        youtube = build("youtube", "v3", developerKey=self.valves.YOUTUBE_API_KEY)
        search_resp = (
            youtube.search()
            .list(part="snippet", q=query, type="video", maxResults=max_results)
            .execute()
        )
        results: List[Dict] = []
        video_ids: List[str] = []
        for item in search_resp.get("items", []):
            vid = item["id"]["videoId"]
            snip = item["snippet"]
            video_ids.append(vid)
            entry = {
                "video_id": vid,
                "title": snip.get("title", ""),
                "channel": snip.get("channelTitle", ""),
                "published_at": snip.get("publishedAt", ""),
                "description": snip.get("description", ""),
            }
            results.append(entry)
        if video_ids:
            detail_resp = (
                youtube.videos()
                .list(part="statistics,contentDetails", id=",".join(video_ids))
                .execute()
            )
            detail_map = {item["id"]: item for item in detail_resp.get("items", [])}
            for entry in results:
                det = detail_map.get(entry["video_id"], {})
                stats = det.get("statistics", {})
                cd = det.get("contentDetails", {})
                entry.update(
                    {
                        "view_count": stats.get("viewCount", "0"),
                        "like_count": stats.get("likeCount", "0"),
                        "comment_count": stats.get("commentCount", "0"),
                        "length": cd.get("duration", ""),
                    }
                )
        return results

    def search(
        self,
        query: str,
        max_results: int = 10,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> Dict:
        """
        Search YouTube for videos matching a keyword, with LRU caching
        and max_results validation.

        Returns a dict matching SearchResult schema.
        """
        entries = self._search_logic(query, max_results)
        # Validate & serialize via Pydantic
        items = [SearchItem(**e) for e in entries]
        result = SearchResult(results=items).dict()
        if __event_emitter__:
            for entry in items:
                __event_emitter__(
                    {"type": "message", "data": {"content": json.dumps(entry.dict())}}
                )
        return result
