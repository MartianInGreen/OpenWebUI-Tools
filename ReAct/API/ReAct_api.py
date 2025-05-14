# ----------------------------------------------------
# Developer: Hannah R. 
# Created: March 4th, 2025
# Lisence: MIT
# Description: API for ReAct in OpenWebUI
# ----------------------------------------------------

# -------------------------------------------------
# Imports
# -------------------------------------------------


import chromadb
import sqlite3

import os
from dotenv import load_dotenv
import hashlib
import uuid
import base64

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.responses import RedirectResponse
from werkzeug.utils import secure_filename as werkzeug_secure_filename
import aiohttp

from lib import image_gen, gpt_image, wolfram, youtube, search, deep_search

# -------------------------------------------------
# Setup
# -------------------------------------------------

STORAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')
# Create folder if it doesn't exist
if not os.path.exists(STORAGE_PATH):
    os.makedirs(STORAGE_PATH)

persistent_client = chromadb.PersistentClient(path=STORAGE_PATH)
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middlewear
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origin_regex=None,
    expose_headers=[],
    max_age=600,
)

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    print(f"OPTIONS request received for path: {full_path}")
    return {"message": "OK"}

# --------------------------------------------------------
# Helper Functions
# --------------------------------------------------------

def check_admin_key(key: str):
    #print(key)
    print(os.getenv('API_KEY'))
    # sha256 the incoming key
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    # Check if the hash matches the admin key
    return key_hash == os.getenv('API_KEY')

def require_api_key(func):
    async def wrapper(request: Request, *args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
            
        key = str(auth_header.split(" ")[1])
        if not check_admin_key(key):
            raise HTTPException(status_code=401, detail="Invalid API key")
            
        return await func(request, *args, **kwargs)
    return wrapper

# --------------------------------------------------------
# API Endpoints
# --------------------------------------------------------

@app.post("/api/v1/imageGen")
@require_api_key
async def imageGen(request: Request):
    data = await request.json()
    
    prompt = data.get('prompt')
    model = data.get('model')
    image_size = data.get('image_size')
    auth_key = os.getenv("FAL_AI_KEY")

    # Check if the api/content dir exists
    if not os.path.exists('content'):
        os.makedirs('content')

    if not prompt or not image_size or not auth_key:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    if model == "pro":
        image = await image_gen.create_image_pro(prompt, image_size, auth_key)
        return image
    if model == "dev":
        image = await image_gen.create_image_basic(prompt, image_size, auth_key)
        return image
    if model == "image-to-image":
        image_url = data.get('image_url')
        image = await image_gen.image_to_image(prompt, image_url, image_size, auth_key)
        return image
    
@app.post("/api/v1/gptImageGen")
@require_api_key
async def gptImageGen(request: Request):
    data = await request.json()
    prompt = data.get('prompt')
    images = data.get('images')
    image_size = data.get('image_size')
    quality = data.get('quality')
    
    # Check if images is a list of base64 encoded images
    if not isinstance(images, list):
        raise HTTPException(status_code=400, detail="Images must be a list of base64 encoded images")
    
    # Decode the images
    decoded_images = [base64.b64decode(image) for image in images]
    
    # Generate the images
    generated_images = await gpt_image.gpt_image_gen(prompt, decoded_images, image_size, quality)
    
    return generated_images

@app.post("/api/v1/wolfram")
@require_api_key
async def wolfram(request: Request):
    data = await request.json()
    query = data.get('query')
    return await wolfram.wolframAlpha(query)

@app.post("/api/v1/youtube")
@require_api_key
async def youtube(request: Request):
    data = await request.json()
    video_id = data.get('video_id')
    return await youtube.youtube_func(video_id)

@app.post("/api/v1/search")
@require_api_key
async def search(request: Request):
    data = await request.json()
    query = data.get('query')
    search_provider = data.get('search_provider')
    
    if search_provider == "perplexity":
        perplexity_type = data.get('perplexity_type')
        return await search.serach_perplexity(query, perplexity_type)
    elif search_provider == "exa":
        return await search.serach_exa(query)
    elif search_provider == "brave":
        # Not implemented yet, return error
        raise HTTPException(status_code=501, detail="Brave search not implemented")
    else:
        raise HTTPException(status_code=400, detail="Invalid search provider")

@app.post("/api/v1/deepSearch")
@require_api_key
async def deepSearch(request: Request):
    data = await request.json()
    query = data.get('query')
    return await deep_search.deep_search_func(query)

@app.post("/api/v1/video_gen")
@require_api_key
async def video_gen(request: Request):
    pass

@app.post("/api/v1/jupyter/upload")
@require_api_key
async def jupyter_upload(request: Request):
    data = await request.json()
    file_path = data.get('file_path')
    file_name = data.get('file_name')
    file_type = data.get('file_type')
    file_data = data.get('file_data')
    return await jupyter.jupyter_upload(file_path, file_name, file_type, file_data)

@app.post("/api/v1/jupyter/storage/{file_path}")
@require_api_key
async def jupyter_storage(request: Request, file_path: str):
    # Get the file content directly instead of redirecting
    url = f"{os.getenv('JUPYTER_URL')}/api/contents/{file_path}?token={os.getenv('JUPYTER_TOKEN')}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=404, detail="File not found")
            
            file_data = await response.json()
            
            # Return the file content directly
            if file_data.get("type") == "file":
                content = file_data.get("content", "")
                format_type = file_data.get("format", "text")
                
                if format_type == "base64":
                    # For binary files, determine the media type based on file extension
                    decoded_content = base64.b64decode(content)
                    file_extension = os.path.splitext(file_path)[1].lower()
                    
                    # Set appropriate media type for common file formats
                    if file_extension in ['.png']:
                        media_type = "image/png"
                    elif file_extension in ['.jpg', '.jpeg']:
                        media_type = "image/jpeg"
                    elif file_extension in ['.gif']:
                        media_type = "image/gif"
                    elif file_extension in ['.svg']:
                        media_type = "image/svg+xml"
                    elif file_extension in ['.webp']:
                        media_type = "image/webp"
                    elif file_extension in ['.pdf']:
                        media_type = "application/pdf"
                    elif file_extension in ['.doc', '.docx']:
                        media_type = "application/msword"
                    elif file_extension in ['.xls', '.xlsx']:
                        media_type = "application/vnd.ms-excel"
                    elif file_extension in ['.ppt', '.pptx']:
                        media_type = "application/vnd.ms-powerpoint"
                    elif file_extension in ['.csv']:
                        media_type = "text/csv"
                    elif file_extension in ['.json']:
                        media_type = "application/json"
                    elif file_extension in ['.xml']:
                        media_type = "application/xml"
                    elif file_extension in ['.zip']:
                        media_type = "application/zip"
                    elif file_extension in ['.mp3']:
                        media_type = "audio/mpeg"
                    elif file_extension in ['.mp4']:
                        media_type = "video/mp4"
                    else:
                        # Default for other binary files
                        media_type = "application/octet-stream"
                    
                    # For images and PDFs, return with appropriate content type for direct display
                    if media_type.startswith("image/") or media_type == "application/pdf":
                        return Response(
                            content=decoded_content,
                            media_type=media_type
                        )
                    else:
                        # For other binary files, return as downloadable content
                        return Response(
                            content=decoded_content,
                            media_type=media_type,
                            headers={"Content-Disposition": f"attachment; filename={os.path.basename(file_path)}"}
                        )
                else:
                    # For text files, return as plain text
                    return Response(content=content, media_type="text/plain")
            else:
                raise HTTPException(status_code=400, detail="Not a file")

@app.post("/api/v1/jupyter/execute")
@require_api_key
async def jupyter(request: Request):
    data = await request.json()
    code = data.get('code')
    return await jupyter.python_code_execution(code)

@app.post("/api/v1/memory")
@require_api_key
async def memory(request: Request):
    pass

@app.get("/content/{filename}")
async def get_content(filename: str):
    # Make filename safe
    safe_filename = werkzeug_secure_filename(filename)
    file_path = os.path.join('api/content', safe_filename)
    
    # Ensure the file is within the api/content directory
    if not os.path.abspath(file_path).startswith(os.path.abspath('api/content')):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

@app.post("/content")
@require_api_key
async def upload_file(request: Request):
    form = await request.form()
    file = form["file"]
    
    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}_{werkzeug_secure_filename(file.filename)}"
    file_path = os.path.join('content', filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Generate the URL
    url = f"{os.getenv('EXTERNAL_URL')}/{filename}"
    
    return JSONResponse(content={"url": url})