import requests
import json
import os
import uuid
import base64
import websockets
import asyncio
from urllib.parse import urljoin

def ensure_directories(directory_path: str) -> bool:
    """
    Recursively ensure that the entire directory_path exists on the Jupyter Notebook server.

    Parameters:
        os.getenv("JUPYTER_URL") (str): Base URL of the Jupyter Notebook server.
        token (str): API token for authentication.
        directory_path (str): Relative directory path (e.g., "outputs" or "subdir/outputs").

    Returns:
        bool: True if all directories exist or were created successfully, False otherwise.
    """
    parts = [p for p in directory_path.strip("/").split("/") if p]
    cumulative = ""

    for part in parts:
        cumulative = f"{cumulative}/{part}" if cumulative else part
        url = f"{os.getenv("JUPYTER_URL")}/api/contents/{cumulative}?token={os.getenv("JUPYTER_TOKEN")}"

        response = requests.get(url)
        if response.status_code == 200:
            continue

        payload = {"type": "directory"}
        headers = {"Content-Type": "application/json"}
        response = requests.put(url, data=json.dumps(payload), headers=headers)

        if response.status_code not in (200, 201):
            print(
                f"Failed to create directory '{cumulative}'. Status code: {response.status_code}"
            )
            print("Response:", response.text)
            return False

    return True

def jupyter_upload(
    file_type: str,
    file_data,
    file_path: str,
    file_name: str,
    already_encoded: bool = False,
):
    """
    Upload a file to a Jupyter Notebook server via its REST API.

    This function ensures that the target directory exists (creating it recursively if needed).
    If the upload is successful, it returns the URL to access the file; otherwise, it returns None.

    Parameters:
        os.getenv("JUPYTER_URL") (str): Base URL of the Jupyter Notebook server (e.g., "http://localhost:8888").
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
        if not ensure_directories(os.getenv("JUPYTER_URL"), os.getenv("JUPYTER_TOKEN"), file_path):
            print("Aborting file upload due to directory creation failure.")
            return None

    full_path = os.path.join(file_path, file_name).replace("\\", "/")
    url = f"{os.getenv("JUPYTER_URL")}/api/contents/{full_path}?token={os.getenv("JUPYTER_TOKEN")}"

    payload = {"type": "file", "format": file_type, "content": ""}

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
        access_url = f"{os.getenv("JUPYTER_URL")}/files/{full_path}?token={os.getenv("JUPYTER_TOKEN")}"
        return access_url
    else:
        print(f"Failed to upload file. Status code: {response.status_code}")
        print("Response:", response.text)
        return None

async def python_code_execution(code: str):
    session = requests.Session()
    headers = {}
    
    timeout = 60
    
    # Construct API URLs with authentication token if provided
    params = f"?token={os.getenv("JUPYTER_TOKEN")}" if os.getenv("JUPYTER_TOKEN") else ""
    kernel_url = urljoin(jupyter_url, f"/api/kernels{params}")

    try:
        response = session.post(
            kernel_url, headers=headers, cookies=session.cookies
        )
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
                    if (
                        message_data.get("parent_header", {}).get("msg_id")
                        == msg_id
                    ):
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
                                    response = jupyter_upload(
                                        os.getenv("JUPYTER_URL_BASE"),
                                        os.getenv("JUPYTER_TOKEN"),
                                        "base64",
                                        data["image/png"],
                                        "outputs",
                                        image_name,
                                        already_encoded=True,
                                    )
                                except Exception as e:
                                    response = f"Error: {str(e)}"

                                # Construct proper URL and append to result
                                result.append(f"![Image]({response})\n")
                            elif "text/plain" in data:
                                result.append(data["text/plain"])

                        elif msg_type == "error":
                            stderr += "\n".join(
                                message_data["content"]["traceback"]
                            )

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
                f"{kernel_url}/{kernel_id}",
                headers=headers,
                cookies=session.cookies,
            )

    return {
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "result": "\n".join(result).strip() if result else "",
    }