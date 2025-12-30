from typing import Any, Optional
import asyncio
import base64
import mimetypes
from pathlib import Path
import tempfile
import uuid
import httpx
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import os
from dotenv import load_dotenv

# Load .env from the project root (where pyproject.toml lives)
# This ensures API keys are always loaded from the server's own .env file,
# regardless of where the MCP client launches from
_SERVER_DIR = Path(__file__).resolve().parent  # src/image_gen/
_PROJECT_ROOT = _SERVER_DIR.parent.parent  # project root
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE)

TOGETHER_AI_BASE = "https://api.together.xyz/v1/images/generations"
API_KEY = os.getenv("TOGETHER_AI_API_KEY")
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"
DEFAULT_IMG2IMG_MODEL = "black-forest-labs/FLUX.1-kontext-dev"
MAX_IMAGE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB (increased for generated images)
MAX_RETURN_IMAGE_BYTES = 4 * 1024 * 1024  # 4MB max for returning image data to client
API_TIMEOUT = 120.0  # 2 minutes for image generation APIs
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# Directory for auto-saving generated images
IMAGE_OUTPUT_DIR = Path(tempfile.gettempdir()) / "mcp-image-gen"
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Replicate API for image upscaling
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
REPLICATE_ESRGAN_VERSION = "42fed1c4974146d4d2414e2be2c5277c7fcf05fcc3a73abf41610695738c1d7b"


def load_image_as_data_uri(image_path: str) -> dict[str, Any]:
    """
    Load a local image file and convert it to a data URI.

    Returns:
        dict with either 'data_uri' key on success or 'error' key on failure
    """
    path = Path(image_path).resolve()

    # Check file exists
    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    if not path.is_file():
        return {"error": f"Path is not a file: {image_path}"}

    # Check file extension
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        return {
            "error": f"Unsupported image format: {suffix}. Supported: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        }

    # Check file size
    file_size = path.stat().st_size
    if file_size > MAX_IMAGE_SIZE_BYTES:
        return {
            "error": f"File too large: {file_size / 1024 / 1024:.2f}MB. Maximum: {MAX_IMAGE_SIZE_BYTES / 1024 / 1024:.0f}MB"
        }

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime_type = mime_map.get(suffix, "image/jpeg")

    # Read and encode file
    try:
        image_data = path.read_bytes()
        b64_data = base64.b64encode(image_data).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{b64_data}"
        return {"data_uri": data_uri}
    except IOError as e:
        return {"error": f"Failed to read image file: {e}"}


def save_image(b64_data: str, prefix: str = "generated") -> str:
    """
    Save a base64-encoded image to the output directory.

    Returns:
        The absolute file path where the image was saved.
    """
    # Generate unique filename
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{unique_id}.png"
    filepath = IMAGE_OUTPUT_DIR / filename

    # Decode and save
    image_bytes = base64.b64decode(b64_data)
    filepath.write_bytes(image_bytes)

    return str(filepath)


server = Server("image-gen")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="generate_image",
            description="""CREATE a new image from a text description. Use this when:
- You need to generate a completely NEW image from scratch
- You have a text prompt describing what to create
- You do NOT have an existing image to modify

OUTPUT: Returns the image AND a file path where it was auto-saved. Use this path directly with edit_image or upscale_image.

DO NOT use this for: editing existing images (use edit_image), or enlarging images (use upscale_image).

Cost: ~$0.003/image at 1024x1024 with FLUX.1-schnell (cheapest), ~$0.04/image with FLUX.1.1-pro (higher quality).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed text description of the image to generate (e.g., 'a golden retriever playing in a sunny park')",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use. Options: 'black-forest-labs/FLUX.1-schnell' (fast, cheap, default), 'black-forest-labs/FLUX.1.1-pro' (higher quality, slower). Falls back to schnell if invalid.",
                    },
                    "width": {
                        "type": "number",
                        "description": "Image width in pixels. For large images, consider generating small (512) then using upscale_image to save costs.",
                    },
                    "height": {
                        "type": "number",
                        "description": "Image height in pixels. For large images, consider generating small (512) then using upscale_image to save costs.",
                    },
                },
                "required": ["prompt", "model"],
            },
        ),
        types.Tool(
            name="edit_image",
            description="""MODIFY an existing image based on text instructions. Use this when:
- You have an EXISTING image that needs to be changed
- You want to transform style (e.g., 'convert to anime style', 'make it look like a painting')
- You want to modify content (e.g., 'change background to sunset', 'add sunglasses')

INPUT: Use the file path returned by generate_image, or provide your own image_path/image_url.
OUTPUT: Returns the edited image AND a file path where it was auto-saved.

DO NOT use this for: creating new images from scratch (use generate_image), or just making images bigger (use upscale_image).
Cost: ~$0.025/image with FLUX.1-kontext-dev, ~$0.04/image with FLUX.1-kontext-pro.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instruction for how to modify the image (e.g., 'convert to watercolor painting', 'add a hat to the person', 'change background to beach')",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Local file path to the source image. Use the path returned by generate_image, or any local image file (<1MB, formats: jpg/png/webp/gif).",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "Public URL to the source image. Only use for permanent URLs (imgur, github raw, etc). Do NOT use temporary/CDN URLs.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model for editing. 'black-forest-labs/FLUX.1-kontext-dev' (default, cheaper) or 'black-forest-labs/FLUX.1-kontext-pro' (better quality).",
                    },
                    "width": {
                        "type": "number",
                        "description": "Output width. Leave empty to match source image.",
                    },
                    "height": {
                        "type": "number",
                        "description": "Output height. Leave empty to match source image.",
                    },
                },
                "required": ["prompt"],
            },
        ),
        types.Tool(
            name="upscale_image",
            description="""ENLARGE an existing image to higher resolution without changing its content. Use this when:
- You need a BIGGER version of an existing image (e.g., 512x512 → 2048x2048)
- You want to create print-quality or banner-sized images cost-effectively
- You generated a small image and need it larger

INPUT: Use the file path returned by generate_image, or provide your own image_path/image_url.
OUTPUT: Returns the upscaled image AND a file path where it was auto-saved.

DO NOT use this for: creating new images (use generate_image), or changing image content/style (use edit_image).
COST-SAVING: Generate 512x512 (~$0.0007) + upscale 4x (~$0.002) = $0.003 total vs $0.16 for direct 2048x2048.
Cost: ~$0.002/image via Replicate.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Local file path to the image to enlarge. Use the path returned by generate_image, or any local image file (<1MB).",
                    },
                    "image_url": {
                        "type": "string",
                        "description": "Public URL to the image. Only use for permanent URLs (imgur, github raw, etc). Do NOT use temporary/CDN URLs.",
                    },
                    "scale": {
                        "type": "integer",
                        "description": "Enlargement factor: 2 (double), 4 (quadruple), or 8. Example: 512px with scale=4 → 2048px. Default: 4.",
                        "minimum": 2,
                        "maximum": 8,
                        "default": 4,
                    },
                    "face_enhance": {
                        "type": "boolean",
                        "description": "Enable GFPGAN to improve facial details. Only use for images containing faces. Default: false.",
                        "default": False,
                    },
                },
            },
        ),
    ]


async def make_together_request(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """Make a request to the Together API with error handling and fallback for incorrect model."""
    request_body = {"model": model, "prompt": prompt, "response_format": "b64_json"}
    headers = {"Authorization": f"Bearer {API_KEY}"}

    if width is not None:
        request_body["width"] = width
    if height is not None:
        request_body["height"] = height

    async def send_request(body: dict) -> (int, dict):
        response = await client.post(TOGETHER_AI_BASE, headers=headers, json=body)
        try:
            data = response.json()
        except Exception:
            data = {}
        return response.status_code, data

    # First request with user-provided model
    status, data = await send_request(request_body)

    # Check if the request failed due to an invalid model error
    if status != 200 and "error" in data:
        error_info = data["error"]
        # Handle error being either a string or a dict
        if isinstance(error_info, dict):
            error_msg = (error_info.get("message") or "").lower()
            error_code = (error_info.get("code") or "").lower()
        else:
            error_msg = str(error_info).lower()
            error_code = ""
        if (
            "model" in error_msg and "not available" in error_msg
        ) or error_code == "model_not_available":
            # Fallback to the default model
            request_body["model"] = DEFAULT_MODEL
            status, data = await send_request(request_body)
            if status != 200 or "error" in data:
                return {
                    "error": f"Fallback API error: {data.get('error', 'Unknown error')} (HTTP {status})"
                }
            return data
        else:
            return {"error": f"Together API error: {error_info}"}
    elif status != 200:
        return {"error": f"HTTP error {status}"}

    return data


async def make_img2img_request(
    client: httpx.AsyncClient,
    prompt: str,
    image_url: str,
    model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """Make an image-to-image request to the Together API."""
    request_body = {
        "model": model,
        "prompt": prompt,
        "image_url": image_url,
        "response_format": "b64_json",
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    if width is not None:
        request_body["width"] = width
    if height is not None:
        request_body["height"] = height

    async def send_request(body: dict) -> tuple[int, dict]:
        response = await client.post(TOGETHER_AI_BASE, headers=headers, json=body)
        try:
            data = response.json()
        except Exception:
            data = {}
        return response.status_code, data

    status, data = await send_request(request_body)

    # Handle model fallback for Kontext models
    if status != 200 and "error" in data:
        error_info = data["error"]
        # Handle error being either a string or a dict
        if isinstance(error_info, dict):
            error_msg = (error_info.get("message") or "").lower()
            error_code = (error_info.get("code") or "").lower()
        else:
            error_msg = str(error_info).lower()
            error_code = ""
        if (
            "model" in error_msg and "not available" in error_msg
        ) or error_code == "model_not_available":
            # Fallback to default image-to-image model
            request_body["model"] = DEFAULT_IMG2IMG_MODEL
            status, data = await send_request(request_body)
            if status != 200 or "error" in data:
                return {
                    "error": f"Fallback API error: {data.get('error', 'Unknown error')} (HTTP {status})"
                }
            return data
        else:
            return {"error": f"Together API error: {error_info}"}
    elif status != 200:
        return {"error": f"HTTP error {status}"}

    return data


async def make_upscale_request(
    client: httpx.AsyncClient,
    image_url: str,
    scale: int = 4,
    face_enhance: bool = False,
) -> dict[str, Any]:
    """Make an upscale request to Replicate's Real-ESRGAN API."""
    if not REPLICATE_API_TOKEN:
        return {"error": "REPLICATE_API_TOKEN environment variable is not set"}

    headers = {
        "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # Sync mode for faster response
    }

    request_body = {
        "version": REPLICATE_ESRGAN_VERSION,
        "input": {
            "image": image_url,
            "scale": scale,
            "face_enhance": face_enhance,
        },
    }

    try:
        # Create prediction
        response = await client.post(
            REPLICATE_API_URL,
            headers=headers,
            json=request_body,
            timeout=120.0,  # Upscaling can take time
        )
        data = response.json()

        if response.status_code == 201 or response.status_code == 200:
            # Check if completed (sync mode)
            if data.get("status") == "succeeded":
                output = data.get("output")
                if output:
                    return {"output_url": output}
                return {"error": "No output URL in response"}

            # If not completed, poll for result
            prediction_url = data.get("urls", {}).get("get")
            if not prediction_url:
                return {"error": "No prediction URL to poll"}

            # Poll for completion
            for _ in range(60):  # Max 60 attempts (2 minutes)
                await asyncio.sleep(2)
                poll_response = await client.get(prediction_url, headers=headers)
                poll_data = poll_response.json()

                status = poll_data.get("status")
                if status == "succeeded":
                    output = poll_data.get("output")
                    if output:
                        return {"output_url": output}
                    return {"error": "No output URL in response"}
                elif status == "failed":
                    error = poll_data.get("error", "Unknown error")
                    return {"error": f"Upscale failed: {error}"}
                elif status == "canceled":
                    return {"error": "Upscale was canceled"}

            return {"error": "Upscale timed out"}
        else:
            error = data.get("detail", data.get("error", f"HTTP {response.status_code}"))
            return {"error": f"Replicate API error: {error}"}

    except httpx.TimeoutException:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": f"Request failed: {e}"}


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can generate images and notify clients of changes.
    """
    if not arguments:
        return [
            types.TextContent(type="text", text="Missing arguments for the request")
        ]

    if name == "generate_image":
        prompt = arguments.get("prompt")
        model = arguments.get("model")
        width = arguments.get("width")
        height = arguments.get("height")

        if not prompt or not model:
            return [
                types.TextContent(type="text", text="Missing prompt or model parameter")
            ]

        try:
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response_data = await make_together_request(
                    client=client,
                    prompt=prompt,
                    model=model,  # User-provided model (or fallback will be used)
                    width=width,
                    height=height,
                )

                if "error" in response_data:
                    return [types.TextContent(type="text", text=response_data["error"])]

                try:
                    b64_image = response_data["data"][0]["b64_json"]
                    # Auto-save the image and return the path
                    saved_path = save_image(b64_image, prefix="generated")

                    # Check if image is too large to return
                    image_bytes = base64.b64decode(b64_image)
                    if len(image_bytes) > MAX_RETURN_IMAGE_BYTES:
                        size_mb = len(image_bytes) / (1024 * 1024)
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Image saved to: {saved_path}\n\nNote: Image is {size_mb:.1f}MB (too large to display inline). Open the file directly to view it.",
                            ),
                        ]

                    return [
                        types.TextContent(
                            type="text",
                            text=f"Image saved to: {saved_path}\nUse this path with edit_image or upscale_image tools.",
                        ),
                        types.ImageContent(
                            type="image", data=b64_image, mimeType="image/png"
                        ),
                    ]
                except (KeyError, IndexError) as e:
                    return [
                        types.TextContent(
                            type="text", text=f"Failed to parse API response: {e}"
                        )
                    ]
        except httpx.TimeoutException:
            return [types.TextContent(type="text", text="Request timed out. Please try again.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error generating image: {e}")]

    elif name == "edit_image":
        prompt = arguments.get("prompt")
        image_url = arguments.get("image_url")
        image_path = arguments.get("image_path")
        model = arguments.get("model", DEFAULT_IMG2IMG_MODEL)
        width = arguments.get("width")
        height = arguments.get("height")

        # Validate prompt
        if not prompt:
            return [types.TextContent(type="text", text="Missing prompt parameter")]

        # Validate image source: exactly one must be provided
        if image_url and image_path:
            return [
                types.TextContent(
                    type="text", text="Provide either image_url or image_path, not both"
                )
            ]

        if not image_url and not image_path:
            return [
                types.TextContent(
                    type="text", text="Must provide either image_url or image_path"
                )
            ]

        # Resolve the image source to a URL (or data URI)
        final_image_url = image_url
        if image_path:
            result = load_image_as_data_uri(image_path)
            if "error" in result:
                return [types.TextContent(type="text", text=result["error"])]
            final_image_url = result["data_uri"]

        # Make the API request
        try:
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response_data = await make_img2img_request(
                    client=client,
                    prompt=prompt,
                    image_url=final_image_url,
                    model=model,
                    width=width,
                    height=height,
                )

                if "error" in response_data:
                    return [types.TextContent(type="text", text=response_data["error"])]

                try:
                    b64_image = response_data["data"][0]["b64_json"]
                    # Auto-save the edited image and return the path
                    saved_path = save_image(b64_image, prefix="edited")

                    # Check if image is too large to return
                    image_bytes = base64.b64decode(b64_image)
                    if len(image_bytes) > MAX_RETURN_IMAGE_BYTES:
                        size_mb = len(image_bytes) / (1024 * 1024)
                        return [
                            types.TextContent(
                                type="text",
                                text=f"Image saved to: {saved_path}\n\nNote: Image is {size_mb:.1f}MB (too large to display inline). Open the file directly to view it.",
                            ),
                        ]

                    return [
                        types.TextContent(
                            type="text",
                            text=f"Image saved to: {saved_path}\nUse this path with edit_image or upscale_image tools.",
                        ),
                        types.ImageContent(
                            type="image", data=b64_image, mimeType="image/png"
                        ),
                    ]
                except (KeyError, IndexError) as e:
                    return [
                        types.TextContent(
                            type="text", text=f"Failed to parse API response: {e}"
                        )
                    ]
        except httpx.TimeoutException:
            return [types.TextContent(type="text", text="Request timed out. Image editing can take up to 2 minutes. Please try again.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error editing image: {e}")]

    elif name == "upscale_image":
        image_url = arguments.get("image_url")
        image_path = arguments.get("image_path")
        scale = arguments.get("scale", 4)
        face_enhance = arguments.get("face_enhance", False)

        # Validate scale
        scale = max(2, min(8, int(scale)))

        # Validate image source: exactly one must be provided
        if image_url and image_path:
            return [
                types.TextContent(
                    type="text", text="Provide either image_url or image_path, not both"
                )
            ]

        if not image_url and not image_path:
            return [
                types.TextContent(
                    type="text", text="Must provide either image_url or image_path"
                )
            ]

        # Resolve the image source to a URL (or data URI)
        final_image_url = image_url
        if image_path:
            result = load_image_as_data_uri(image_path)
            if "error" in result:
                return [types.TextContent(type="text", text=result["error"])]
            final_image_url = result["data_uri"]

        # Make the upscale request
        try:
            async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
                response_data = await make_upscale_request(
                    client=client,
                    image_url=final_image_url,
                    scale=scale,
                    face_enhance=face_enhance,
                )

                if "error" in response_data:
                    return [types.TextContent(type="text", text=response_data["error"])]

                # Fetch the upscaled image from the output URL
                output_url = response_data.get("output_url")
                if not output_url:
                    return [
                        types.TextContent(type="text", text="No output URL in response")
                    ]

                # Download the upscaled image
                img_response = await client.get(output_url, timeout=60.0)
                if img_response.status_code != 200:
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Failed to download upscaled image: HTTP {img_response.status_code}",
                        )
                    ]

                # Convert to base64
                b64_image = base64.b64encode(img_response.content).decode("utf-8")

                # Auto-save the upscaled image and return the path
                saved_path = save_image(b64_image, prefix="upscaled")

                # Check if image is too large to return (would crash client)
                image_size = len(img_response.content)
                if image_size > MAX_RETURN_IMAGE_BYTES:
                    # Image too large - return only path, not image data
                    size_mb = image_size / (1024 * 1024)
                    return [
                        types.TextContent(
                            type="text",
                            text=f"Image saved to: {saved_path}\n\nNote: Upscaled image is {size_mb:.1f}MB (too large to display inline). Open the file directly to view it.",
                        ),
                    ]

                return [
                    types.TextContent(
                        type="text",
                        text=f"Image saved to: {saved_path}\nUse this path with edit_image or upscale_image tools.",
                    ),
                    types.ImageContent(
                        type="image", data=b64_image, mimeType="image/png"
                    ),
                ]
        except httpx.TimeoutException:
            return [types.TextContent(type="text", text="Request timed out. Upscaling can take up to 2 minutes. Please try again.")]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Error upscaling image: {e}")]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="image-gen",
                server_version="0.3.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
