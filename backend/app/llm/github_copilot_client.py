import aiohttp
import json
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

from app.llm.base import LLMClient
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubCopilotClient(LLMClient):
    """Client for GitHub Models (Copilot) API."""

    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = "https://models.github.ai",
        org: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        super().__init__(model, api_key, base_url, embedding_model)
        self.api_key = api_key or settings.GITHUB_API_TOKEN
        if not self.api_key:
            raise ValueError("GitHub API token is required")
        self.org = org
        if not self.base_url:
            self.base_url = "https://models.github.ai"

    def _build_url(self) -> str:
        if self.org:
            return f"{self.base_url}/orgs/{self.org}/inference/chat/completions"
        return f"{self.base_url}/inference/chat/completions"

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        url = self._build_url()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/vnd.github+json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        start_time = time.time()
        if stream:
            return self._stream_response(url, headers, payload, start_time)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"GitHub API error: {error_text}")
                    raise Exception(f"GitHub API error: {response.status} - {error_text}")
                result = await response.json()
                tokens = result.get("usage", {}).get("completion_tokens", 0)
                tokens_per_second = self.calculate_tokens_per_second(start_time, tokens)
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "model": self.model,
                    "provider": "github_copilot",
                    "tokens": tokens,
                    "tokens_per_second": tokens_per_second,
                }

    async def _stream_response(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        start_time: float,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                await self._handle_non_200_response(response)
                content = ""
                token_count = 0
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        if line == "data: [DONE]":
                            break
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line: {line}")
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    delta_content = delta.get("content", "")
                    if delta_content:
                        content += delta_content
                        token_count += 1
                        tokens_per_second = self.calculate_tokens_per_second(start_time, token_count)
                        yield {
                            "content": content,
                            "model": self.model,
                            "provider": "github_copilot",
                            "tokens": token_count,
                            "tokens_per_second": tokens_per_second,
                            "done": False,
                            "timestamp": time.time(),
                        }
                        await asyncio.sleep(0)
                tokens_per_second = self.calculate_tokens_per_second(start_time, token_count)
                yield {
                    "content": content,
                    "model": self.model,
                    "provider": "github_copilot",
                    "tokens": token_count,
                    "tokens_per_second": tokens_per_second,
                    "done": True,
                }

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.warning("GitHub Copilot client does not support embeddings directly.")
        raise NotImplementedError("GitHub Copilot embeddings are not supported")
