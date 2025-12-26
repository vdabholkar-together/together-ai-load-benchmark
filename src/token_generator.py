"""
Generate prompts with specific token counts for testing.
"""
import tiktoken
from typing import List


class TokenGenerator:
    """Generate text with specific token counts."""

    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token generator.
        
        Args:
            model: Model name for tokenizer (default: gpt-4)
        """
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4)
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def generate_prompt(self, target_tokens: int, base_prompt: str = "") -> str:
        """
        Generate a prompt with approximately target_tokens tokens.
        
        Args:
            target_tokens: Target number of tokens
            base_prompt: Base prompt to include
            
        Returns:
            Prompt text with approximately target_tokens tokens
        """
        if target_tokens <= 0:
            return base_prompt

        base_tokens = self.count_tokens(base_prompt) if base_prompt else 0
        remaining_tokens = max(0, target_tokens - base_tokens)

        if remaining_tokens == 0:
            return base_prompt

        # Generate filler text to reach target token count
        # Use a repeating pattern that's token-efficient
        filler_pattern = (
            "This is a test prompt designed to reach a specific token count. "
            "The content here is used to pad the prompt to the desired length. "
            "We repeat this pattern multiple times to achieve the target token count. "
        )

        # Calculate how many times to repeat the pattern
        pattern_tokens = self.count_tokens(filler_pattern)
        if pattern_tokens == 0:
            # Fallback: use single words
            filler_pattern = "test "
            pattern_tokens = self.count_tokens(filler_pattern)

        if pattern_tokens > 0:
            repetitions = max(1, remaining_tokens // pattern_tokens)
            filler_text = filler_pattern * repetitions
        else:
            filler_text = "test " * remaining_tokens

        # Adjust to get closer to target
        current_tokens = self.count_tokens(base_prompt + filler_text)
        if current_tokens < target_tokens:
            # Add more text
            additional_needed = target_tokens - current_tokens
            additional_text = "x " * additional_needed
            filler_text += additional_text
        elif current_tokens > target_tokens:
            # Trim text (approximate)
            excess = current_tokens - target_tokens
            # Remove words approximately
            words = filler_text.split()
            words_to_remove = min(len(words), excess)
            filler_text = " ".join(words[:-words_to_remove]) if words_to_remove > 0 else ""

        final_prompt = base_prompt + "\n\n" + filler_text if base_prompt else filler_text
        return final_prompt.strip()

    def generate_output_prompt(self, target_output_tokens: int, base_prompt: str) -> tuple[str, int]:
        """
        Generate a prompt that will result in approximately target_output_tokens in the response.
        
        This is done by setting max_tokens parameter, but we also need to ensure
        the prompt encourages longer responses.
        
        Args:
            target_output_tokens: Target number of output tokens
            base_prompt: Base prompt
            
        Returns:
            Tuple of (prompt, max_tokens)
        """
        # Add instruction to generate long response
        instruction = f"\n\nPlease provide a detailed response with approximately {target_output_tokens} tokens. "
        prompt = base_prompt + instruction
        
        return prompt, target_output_tokens

