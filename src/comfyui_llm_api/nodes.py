import os
import requests
import base64
from PIL import Image
import io
import numpy as np
import traceback


class LLMAPINode:
    """
    A node that makes calls to OpenAI-compatible LLM APIs.

    Takes an image and prompt as input, along with API configuration,
    and returns the LLM response.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image:"}),
                "base_url": ("STRING", {"default": "https://openrouter.ai/api/v1/chat/completions", "multiline": False}),
                "api_key": (
                    "STRING",
                    {
                        "default": os.getenv("OPENAI_API_KEY", ""),
                        "multiline": False,
                    },
                ),
                "model": ("STRING", {"default": "google/gemini-2.0-flash-001", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1, "round": 0.1, "display": "slider"}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "LLM"

    def process(self, prompt, base_url, model, temperature, api_key, image=None):
        """
        Process the prompt through the LLM API, optionally including an image
        """
        # Validate API key
        if not api_key:
            return ("Error: API key not provided",)

        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openrouter.ai/",  # Required by OpenRouter
                "X-Title": "ComfyUI LLM API Node",  # Required by OpenRouter
            }

            if image is not None:
                # Handle image + text request
                # Convert tensor to numpy array and remove batch dimension
                image_np = (image.cpu().numpy()[0] * 255).astype(np.uint8)
                if image_np.shape[0] == 3:  # If image is in CHW format
                    image_np = np.transpose(image_np, (1, 2, 0))  # Convert to HWC format

                image_pil = Image.fromarray(image_np)
                buffer = io.BytesIO()
                image_pil.save(buffer, format="PNG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                message_content = {"type": "text", "text": prompt, "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                messages = [{"role": "user", "content": [message_content]}]
            else:
                # Handle text-only request
                messages = [{"role": "user", "content": prompt}]

            data = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Make the API call
            response = requests.post(base_url, headers=headers, json=data)

            # Handle the response
            if response.status_code == 200:
                try:
                    result = response.json()
                    return (result["choices"][0]["message"]["content"],)
                except (ValueError, KeyError) as e:
                    error_msg = f"""Failed to parse API response:
Status code: {response.status_code}
Response text: {response.text}
Error: {str(e)}"""
                    print(error_msg)
                    return ("Error: Failed to parse API response. See console for details.",)
            else:
                error_msg = f"""API call failed:
Status code: {response.status_code}
Response: {response.text}
Request URL: {base_url}
Request data: {data}"""
                print(error_msg)
                return (f"Error: API call failed with status {response.status_code}. See console for details.",)

        except Exception as e:
            error_msg = f"""Exception occurred during API call:
Error: {str(e)}
Traceback:
{traceback.format_exc()}
Request URL: {base_url}
Request data: {data}"""
            print(error_msg)
            return ("Error: Exception during API call. See console for details.",)


# Register the node
NODE_CLASS_MAPPINGS = {"LLMAPINode": LLMAPINode}

NODE_DISPLAY_NAME_MAPPINGS = {"LLMAPINode": "LLM API"}
