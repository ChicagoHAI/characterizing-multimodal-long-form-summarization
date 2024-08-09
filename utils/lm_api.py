import hashlib
import os
import pickle
from typing import Any, Optional

import anthropic
from openai import OpenAI
import cohere

OPENAI_KEY = "YOUR API KEY"
ANTHROPIC_KEY = "YOUR API KEY"
COHERE_KEY = "YOUR API KEY"

class LanguageModelAPI:
    def __init__(
        self,
        api_type: str,
        model: str,
        cache_dir: str,
        openai_system_content: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key
        self.api_type = api_type.lower()
        self.cache_dir = cache_dir
        self.model = model
        os.makedirs(cache_dir, exist_ok=True)

        if self.api_type == "claude":
            claude_api_key = self.api_key or ANTHROPIC_KEY
            self.client = anthropic.Client(api_key=claude_api_key)
            assert self.model in [
                "claude-v1",
                "claude-v1.2",
                "claude-v1.3",
                "claude-v1.4",
                "claude-2.0",
                "claude-2.1",
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307"
            ]
        elif self.api_type == "openai":
            # OpenAI.api_key = self.api_key or OPENAI_KEY
            self.client = OpenAI(api_key=self.api_key or OPENAI_KEY)
            self.open_system_content = (
                openai_system_content or "You are a helpful assistant."
            )
            assert self.model in ["gpt-3.5-turbo", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-1106-preview"]
        elif self.api_type == 'cohere':
            cohere_api_key = self.api_key or COHERE_KEY
            self.client = cohere.Client(cohere_api_key)
        else:
            raise ValueError("Invalid API type. Choose either 'claude', 'openai' or 'cohere.")

    def _cache_key(self, prompt: str) -> str:
        return hashlib.sha1(prompt.encode()).hexdigest()

    def _cache_file_path(self, prompt: str) -> str:
        cache_key = self._cache_key(prompt)
        api_dir = os.path.join(self.cache_dir, self.api_type)
        model_dir = os.path.join(api_dir, self.model)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, f"{cache_key}.pkl")

    def _load_from_cache(self, prompt: str) -> Optional[Any]:
        cache_file = self._cache_file_path(prompt)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_to_cache(self, prompt: str, response: Any) -> None:
        cache_file = self._cache_file_path(prompt)
        with open(cache_file, "wb") as f:
            pickle.dump(response, f)

    def chat(self, prompt: str, system_prompt: str = None, max_tokens: int = 4096) -> str:
        response = self._get_response(prompt, system_prompt, max_tokens)
        if self.api_type == "claude":
            return response
        elif self.api_type == "openai":
            print(response)
            return response.choices[0].message.content.strip()
        elif self.api_type == "cohere":
            return response.summary

    def _get_response(self, prompt: str, system_prompt: str = '', max_tokens: int = 4096) -> Any:
        load_index = system_prompt+prompt if system_prompt is not None else prompt
        cached_response = self._load_from_cache(load_index)
        if cached_response is not None:
            return cached_response

        if self.api_type == "claude":
            if self.model == "claude-2.0" or self.model=="claude-2.1":
                response = self._get_response_anthropic(prompt, max_tokens)
            else:
                response = self._get_message_anthropic(prompt, system_prompt, max_tokens)
        elif self.api_type == "openai":
            response = self._get_response_openai(prompt)
        elif self.api_type == "cohere":
            response = self._get_response_cohere(prompt)

        self._save_to_cache(load_index, response)
        return response

    def _get_response_anthropic(self, prompt: str, max_tokens: int) -> Any:

        response = self.client.completions.create(
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens_to_sample=max_tokens,
        )
        return response.completion

    def _get_message_anthropic(self, prompt: str, system_prompt: str, max_tokens: int) -> Any:

        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        return message.content[0].text

    def _get_response_openai(self, prompt: str) -> Any:
        message_list = [
            {"role": "system", "content": self.open_system_content},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_list,
        )
        return response
    
    def _get_response_cohere(self, prompt: str) -> Any:

        response = self.client.summarize(
            text=prompt,
            length='long',
            format='bullets',
            model=self.model,
            extractiveness='high',
            additional_command='',
            temperature=0.3,
            )
        return response