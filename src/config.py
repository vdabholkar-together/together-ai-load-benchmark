"""
Configuration management for concurrency tests.
"""
import os
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TestConfig:
    """Configuration for a concurrency test run."""
    model: str
    concurrency_levels: List[int]
    input_token_counts: List[int]
    output_token_counts: List[int]
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7

    def __post_init__(self):
        """Set default API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.environ.get("TOGETHER_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "TOGETHER_API_KEY not found in environment variables. "
                    "Please set it or provide via config."
                )

    @classmethod
    def from_args(
        cls,
        model: str,
        concurrency: str,
        input_tokens: str,
        output_tokens: str,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> "TestConfig":
        """
        Create config from command-line arguments.
        
        Args:
            model: Model name
            concurrency: Comma-separated concurrency levels (e.g., "32,64,128")
            input_tokens: Comma-separated input token counts (e.g., "1000,10000")
            output_tokens: Comma-separated output token counts (e.g., "1000,10000")
            api_key: Optional API key (defaults to env var)
            max_tokens: Optional max tokens parameter
            temperature: Temperature for generation
        """
        concurrency_levels = [int(x.strip()) for x in concurrency.split(",")]
        input_token_counts = [int(x.strip()) for x in input_tokens.split(",")]
        output_token_counts = [int(x.strip()) for x in output_tokens.split(",")]

        return cls(
            model=model,
            concurrency_levels=concurrency_levels,
            input_token_counts=input_token_counts,
            output_token_counts=output_token_counts,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
        )

