# core/ollama_client.py

import httpx
import json
from typing import Union, List, Dict, Optional, Generator

class OllamaClient:
    """
    Client for interacting with the Ollama API synchronously.
    Handles text generation and embeddings.
    """
    def __init__(self, base_url: str = "http://localhost:11434/api"):
        self.base_url = base_url
        self.sync_client = httpx.Client(base_url=self.base_url, timeout=300.0)

    def generate_text(self, model_name: str,
                      prompt: Optional[str] = None,
                      messages: Optional[List[Dict[str, str]]] = None,
                      system_message: Optional[str] = None,
                      temperature: float = 0.7, max_tokens: int = 2048,
                      stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generates text using a specified Ollama model synchronously.
        Prioritizes 'messages' for /chat endpoint, falls back to 'prompt' for /generate.
        """
        headers = {"Content-Type": "application/json"}
        request_data = {
            "model": model_name,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": stream
        }

        if messages:
            url_path = "/chat"
            request_data["messages"] = messages
        elif prompt:
            url_path = "/generate"
            request_data["prompt"] = prompt
            if system_message:
                request_data["prompt"] = f"{system_message}\n\n{prompt}"
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided to generate_text.")

        try:
            if stream:
                with self.sync_client.stream("POST", url_path, headers=headers, json=request_data, timeout=None) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            try:
                                # REMOVED .decode('utf-8') here
                                json_data = json.loads(line) 
                                content = self._extract_content(json_data)
                                if content is not None:
                                    yield content
                                if json_data.get('done'):
                                    break
                            except json.JSONDecodeError:
                                continue
            else:
                response = self.sync_client.post(url_path, headers=headers, json=request_data)
                response.raise_for_status()
                result = response.json()
                return self._extract_content(result)

        except httpx.ConnectError:
            error_msg = "Error: Could not connect to Ollama server. Is it running?"
            print(f"{{OllamaClient}} {error_msg}")
            return error_msg if not stream else None
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            print(f"{{OllamaClient}} {error_msg}")
            return error_msg if not stream else None
        except ValueError as e:
            error_msg = f"{{OllamaClient}} Configuration Error: {e}"
            print(error_msg)
            return error_msg if not stream else None
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(f"{{OllamaClient}} {error_msg}")
            return error_msg if not stream else None

    def _extract_content(self, response: dict) -> Optional[str]:
        if 'message' in response and 'content' in response['message']:
            return response['message']['content']
        elif 'response' in response:
            return response['response']
        return None

    def get_embeddings(self, model_name: str, text: str) -> Optional[List[float]]:
        url_path = "/embeddings"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model_name,
            "prompt": text
        }
        try:
            response = self.sync_client.post(url_path, headers=headers, json=data)
            response.raise_for_status()
            return response.json().get('embedding')
        except httpx.ConnectError:
            print("Error: Could not connect to Ollama server. Is it running?")
        except httpx.HTTPStatusError as e:
            print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None
