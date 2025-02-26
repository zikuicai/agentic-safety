import json

from typing import Optional, Any, Dict, List, Union
import json
from enum import Enum
import re

CUSTOM_LLM_PROVIDERS = [
    "openai",
    "azure",
    "xinference",
    "voyage",
    "mistral",
    "custom_openai",
    "triton",
    "anyscale",
    "openrouter",
    "deepinfra",
    "perplexity",
    "groq",
    "nvidia_nim",
    "cerebras",
    "sambanova",
    "ai21_chat",
    "volcengine",
    "deepseek",
    "fireworks_ai",
    "ollama",
    "vertex_ai",
    "gemini",
    "databricks",
    "watsonx",
    "cohere",
    "huggingface",
    "bedrock",
    "azure_ai",
    "together_ai",
    "openai_like",
    "jina_ai",
]


class RobustJSONParser:
    @staticmethod
    def _fix_quotes(json_str: str) -> str:
        """Convert single quotes to double quotes and ensure property names are quoted."""
        # First, handle special cases where we don't want to replace quotes
        # Save string literals with escaped quotes
        saved_strings = []

        def save_string(match):
            saved_strings.append(match.group(0))
            return f"__STRING_PLACEHOLDER_{len(saved_strings) - 1}__"

        # Save correctly formatted string literals
        pattern = r'"(?:[^"\\]|\\.)*"'
        json_str = re.sub(pattern, save_string, json_str)

        # Fix property names that aren't quoted
        json_str = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)

        # Convert remaining single quotes to double quotes
        json_str = json_str.replace("'", '"')

        # Restore saved strings
        for i, saved in enumerate(saved_strings):
            json_str = json_str.replace(f"__STRING_PLACEHOLDER_{i}__", saved)

        return json_str

    @staticmethod
    def _fix_values(json_str: str) -> str:
        """Fix unquoted string values."""
        # Fix unquoted string values that are letters A-Z or words
        json_str = re.sub(r':\s*([A-Za-z]+)\s*(,|}|$)', r': "\1"\2', json_str)

        # Fix boolean values
        json_str = re.sub(r':\s*True\s*(,|}|$)', r': true\1', json_str)
        json_str = re.sub(r':\s*False\s*(,|}|$)', r': false\1', json_str)

        # Fix None/null values
        json_str = re.sub(r':\s*None\s*(,|}|$)', r': null\1', json_str)

        return json_str

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract JSON-like content from text."""
        # Try to find content between outermost curly braces
        matches = re.findall(r'({[^{}]*(?:{[^{}]*})*[^{}]*})', text)
        if matches:
            return matches[0]
        return None

    @staticmethod
    def _fix_trailing_commas(json_str: str) -> str:
        """Remove trailing commas in objects and arrays."""
        # Fix objects with trailing commas
        json_str = re.sub(r',(\s*})', r'\1', json_str)
        # Fix arrays with trailing commas
        json_str = re.sub(r',(\s*\])', r'\1', json_str)
        return json_str

    @classmethod
    def parse(cls, text: str, debug: bool = False) -> Union[Dict, Any]:
        """
        Parse text into a JSON object, applying multiple fixes for common issues.
        
        Args:
            text: The text to parse
            debug: If True, print intermediate steps
            
        Returns:
            Parsed JSON object
        """
        if debug:
            print("Original text:", text)

        # Extract JSON if embedded in other text
        json_str = cls._extract_json(text) or text
        if debug:
            print("Extracted JSON:", json_str)

        # Apply fixes
        json_str = cls._fix_quotes(json_str)
        if debug:
            print("After fixing quotes:", json_str)

        json_str = cls._fix_values(json_str)
        if debug:
            print("After fixing values:", json_str)

        json_str = cls._fix_trailing_commas(json_str)
        if debug:
            print("After fixing trailing commas:", json_str)

        # Final cleanup
        json_str = json_str.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if debug:
                print(f"Failed to parse JSON: {str(e)}")
            raise


parser = RobustJSONParser()


def get_response_content(response, to_json=False):
    out = response.choices[0].message.content
    if to_json:
        out = json.loads(out)
        # out = parser.parse(out)
    return out


def get_response_cost(response):
    return response._hidden_params["response_cost"]


def remove_provider_name_from_model(model_name: str):
    for provider in CUSTOM_LLM_PROVIDERS:
        if model_name.startswith(provider):
            return model_name[len(provider):].strip().strip("/").strip('\\')
    return model_name
