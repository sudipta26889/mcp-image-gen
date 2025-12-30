# Image Generation MCP Server

A Model Context Protocol (MCP) server that enables seamless generation of high-quality images via Together AI. This server provides a standardized interface to specify image generation parameters.

<a href="https://glama.ai/mcp/servers/o0137xiz62">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/o0137xiz62/badge" alt="Image Generation Server MCP server" />
</a>

## Features

- High-quality image generation powered by the Flux.1 Schnell model
- Image-to-image editing using FLUX.1 Kontext models
- Image upscaling (2x-8x) using Real-ESRGAN via Replicate
- Support for customizable dimensions (width and height)
- Accept images via URL or local file path
- Cost-effective large image workflow: generate small, then upscale
- Clear error handling for prompt validation and API issues
- Easy integration with MCP-compatible clients

## Installation

### Step 1: Set Up API Keys

Create a `.env` file in the project root directory:

```bash
cd /path/to/mcp-image-gen
touch .env
```

Add your API keys to the `.env` file:

```
TOGETHER_AI_API_KEY=your_together_ai_api_key_here
REPLICATE_API_TOKEN=your_replicate_api_token_here
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `TOGETHER_AI_API_KEY` | Yes | For `generate_image` and `edit_image` tools |
| `REPLICATE_API_TOKEN` | Optional | For `upscale_image` tool only |

> **Note:** The server automatically loads API keys from the `.env` file in its own directory. You don't need to pass them via MCP client configuration.

### Step 2: Configure Your MCP Client

Choose the configuration method for your MCP client:

---

#### Claude Desktop (JSON)

Config file location:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "image-gen": {
      "command": "uv",
      "args": ["--directory", "/path/to/mcp-image-gen", "run", "image-gen"]
    }
  }
}
```

---

#### Claude Code CLI

```bash
claude mcp add image-gen -- uv --directory /path/to/mcp-image-gen run image-gen
```

To add to a specific scope:

```bash
# Project-level (stored in .mcp.json in current directory)
claude mcp add image-gen -s project -- uv --directory /path/to/mcp-image-gen run image-gen

# User-level (available across all projects)
claude mcp add image-gen -s user -- uv --directory /path/to/mcp-image-gen run image-gen
```

---

#### OpenAI Codex (TOML)

Config file location: `~/.codex/config.toml`

```toml
[mcp_servers.image-gen]
command = ["uv", "--directory", "/path/to/mcp-image-gen", "run", "image-gen"]
```

> **Important:** The section must be named `mcp_servers` (with underscore). Using `mcp-servers` will cause Codex to silently ignore the configuration.

## Available Tools

The server implements three tools:

### generate_image

Generates an image based on the given textual prompt and optional dimensions.

**Input Schema:**

```json
{
  "prompt": {
    "type": "string",
    "description": "A descriptive prompt for generating the image (e.g., 'a futuristic cityscape at sunset')"
  },
  "model": {
    "type": "string",
    "description": "The exact model name as it appears in Together AI. If incorrect, it will fallback to the default model (black-forest-labs/FLUX.1-schnell)."
  },
  "width": {
    "type": "integer",
    "description": "Width of the generated image in pixels (optional)"
  },
  "height": {
    "type": "integer",
    "description": "Height of the generated image in pixels (optional)"
  }
}
```

### edit_image

Edits or transforms an existing image using AI-powered image-to-image generation.

**Input Schema:**

```json
{
  "prompt": {
    "type": "string",
    "description": "Text instruction describing how to edit/transform the image (e.g., 'make the background a sunset', 'convert to anime style')"
  },
  "image_url": {
    "type": "string",
    "description": "URL to the source image (mutually exclusive with image_path)"
  },
  "image_path": {
    "type": "string",
    "description": "Local file path to the source image (mutually exclusive with image_url)"
  },
  "model": {
    "type": "string",
    "description": "The model for image editing (default: black-forest-labs/FLUX.1-kontext-dev)"
  },
  "width": {
    "type": "integer",
    "description": "Width of the output image in pixels (optional)"
  },
  "height": {
    "type": "integer",
    "description": "Height of the output image in pixels (optional)"
  }
}
```

**Example Usage:**

- Edit by URL: `{"prompt": "make it sunset", "image_url": "https://example.com/photo.jpg"}`
- Edit by local file: `{"prompt": "convert to anime style", "image_path": "/path/to/image.png"}`

**Note:** Local file images must be under 1MB and in a supported format (jpg, jpeg, png, webp, gif).

### upscale_image

Upscales an image 2x-8x using Real-ESRGAN via Replicate. Cost-effective for large images.

**Input Schema:**

```json
{
  "image_url": {
    "type": "string",
    "description": "URL to the image to upscale (mutually exclusive with image_path)"
  },
  "image_path": {
    "type": "string",
    "description": "Local file path to the image to upscale (mutually exclusive with image_url)"
  },
  "scale": {
    "type": "integer",
    "description": "Upscale factor: 2, 4, or 8. Default: 4 (e.g., 512x512 → 2048x2048)"
  },
  "face_enhance": {
    "type": "boolean",
    "description": "Apply GFPGAN face enhancement for better facial details. Default: false"
  }
}
```

**Example Usage:**

- Upscale 4x: `{"image_url": "https://example.com/small.jpg", "scale": 4}`
- With face enhancement: `{"image_path": "/path/to/portrait.png", "scale": 2, "face_enhance": true}`

**Cost:** ~$0.002 per image via Replicate

## Cost-Saving Workflow

For large images (e.g., 2048×2048), generate small and upscale:

1. **Generate small** with `generate_image` at 512×512 using FLUX.1 schnell (~$0.0007)
2. **Upscale 4x** with `upscale_image` to get 2048×2048 (~$0.002)
3. **Total: ~$0.003** vs ~$0.16 for direct high-res generation (53x cheaper!)

## Prerequisites

- Python 3.12 or higher
- httpx
- mcp

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch (`feature/my-new-feature`)
3. Commit your changes
4. Push the branch to your fork
5. Open a Pull Request

For significant changes, please open an issue first to discuss your proposed changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.