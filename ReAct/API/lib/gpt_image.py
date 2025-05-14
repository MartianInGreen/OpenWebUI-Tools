import base64, os, uuid, file
from openai import OpenAI

def gpt_image_save(base64_img: str):
    # Generate a unique filename
    filename = f"{uuid.uuid4().hex}_imagegpt.png"
    file_path = os.path.join('content', filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(file.read())
    
    # Generate the URL
    url = f"{os.getenv('EXTERNAL_URL')}/{filename}"
    
    # Return the URL
    return url

def gpt_image_gen(prompt: str, image_data: str, image_size: str, quality: str):
    # Check if image_size and quality are valid
    image_sizes = ["1024x1024", "1536x1024", "1024x1536"]
    qualities = ["low", "medium", "high"]
    if image_size not in image_sizes:
        return {"success": False, "data": None, "iframe": False, "failure": "Invalid image size"}
    if quality not in qualities:
        return {"success": False, "data": None, "iframe": False, "failure": "Invalid quality"}
    
    if image_data: 
        prompt = prompt + "\nPlease edit the provided image(s)."
        
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        if not image_data:
            images = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                size=image_size,
                quality=quality,
                moderation="low",
            )
            
            urls = []
            
            # Save the image to a file
            for i, img in enumerate(images.data, 1):
                image_url = gpt_image_save(img.b64_json)
                urls.append(image_url)
            
            return {"success": True, "data": images, "iframe": False}  
        else:
            images = client.images.generate(
                model="gpt-image-1",
                prompt=prompt,
                n=1,
                size=image_size,
                quality=quality,
                moderation="low",
                image=image_data,
            )
            
            urls = []
            
            # Save the image to a file
            for i, img in enumerate(images.data, 1):
                image_url = gpt_image_save(img.b64_json)
                urls.append(image_url)
               
            #for i, img in enumerate(images.data, 1):
            #    yield f"![image_{i}](data:image/png;base64,{img.b64_json})"
            
            return {"success": True, "data": images, "iframe": False}   
    except Exception as e:
        return {"success": False, "data": None, "iframe": False, "failure": "No images returned"}