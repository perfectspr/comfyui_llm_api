"""Top-level package for comfyui_llm_api."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Peter Sun"""
__email__ = "pengrui.sun@outlook.com"
__version__ = "0.0.1"

from .src.comfyui_llm_api.nodes import NODE_CLASS_MAPPINGS
from .src.comfyui_llm_api.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
