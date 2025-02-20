#!/usr/bin/env python

"""Tests for `comfyui_llm_api` package."""

import pytest
import responses
import torch
import numpy as np
from PIL import Image
import io
import base64
from src.comfyui_llm_api.nodes import LLMAPINode


@pytest.fixture
def llm_node():
    """Fixture to create an LLMAPINode instance."""
    return LLMAPINode()


@pytest.fixture
def mock_image():
    """Create a mock torch tensor image."""
    # Create a 3x64x64 tensor with random values between 0 and 1
    return torch.rand(1, 3, 64, 64)


def test_llm_node_initialization(llm_node):
    """Test that the node can be instantiated."""
    assert isinstance(llm_node, LLMAPINode)


def test_input_types():
    """Test the node's input metadata."""
    input_types = LLMAPINode.INPUT_TYPES()

    assert "required" in input_types
    assert "optional" in input_types

    required = input_types["required"]
    assert "prompt" in required
    assert "base_url" in required
    assert "api_key" in required
    assert "model" in required
    assert "temperature" in required

    optional = input_types["optional"]
    assert "image" in optional


def test_return_types():
    """Test the node's return metadata."""
    assert LLMAPINode.RETURN_TYPES == ("STRING",)
    assert LLMAPINode.RETURN_NAMES == ("response",)
    assert LLMAPINode.FUNCTION == "process"
    assert LLMAPINode.CATEGORY == "LLM"


def test_missing_api_key(llm_node):
    """Test handling of missing API key."""
    result = llm_node.process(prompt="Test prompt", base_url="https://api.example.com", model="test-model", temperature=0.7, api_key="")
    assert result == ("Error: API key not provided",)


@responses.activate
def test_successful_text_request(llm_node):
    """Test successful text-only API request."""
    # Mock the API response
    responses.add(responses.POST, "https://api.example.com", json={"choices": [{"message": {"content": "Test response"}}]}, status=200)

    result = llm_node.process(
        prompt="Test prompt", base_url="https://api.example.com", model="test-model", temperature=0.7, api_key="test-key"
    )

    assert result == ("Test response",)
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["Authorization"] == "Bearer test-key"


@responses.activate
def test_successful_image_request(llm_node, mock_image):
    """Test successful image+text API request."""
    # Mock the API response
    responses.add(
        responses.POST, "https://api.example.com", json={"choices": [{"message": {"content": "Image description response"}}]}, status=200
    )

    result = llm_node.process(
        prompt="Describe this image:",
        base_url="https://api.example.com",
        model="test-model",
        temperature=0.7,
        api_key="test-key",
        image=mock_image,
    )

    assert result == ("Image description response",)
    assert len(responses.calls) == 1

    # Verify the request contains image data
    request_body = responses.calls[0].request.body.decode()
    assert "data:image/png;base64" in request_body


@responses.activate
def test_api_error_handling(llm_node):
    """Test handling of API errors."""
    # Mock an API error response
    responses.add(responses.POST, "https://api.example.com", json={"error": "Invalid request"}, status=400)

    result = llm_node.process(
        prompt="Test prompt", base_url="https://api.example.com", model="test-model", temperature=0.7, api_key="test-key"
    )

    assert result[0].startswith("Error: API call failed")


@responses.activate
def test_malformed_response_handling(llm_node):
    """Test handling of malformed API responses."""
    # Mock a malformed response
    responses.add(responses.POST, "https://api.example.com", json={"invalid": "response format"}, status=200)

    result = llm_node.process(
        prompt="Test prompt", base_url="https://api.example.com", model="test-model", temperature=0.7, api_key="test-key"
    )

    assert result[0].startswith("Error: Failed to parse API response")
