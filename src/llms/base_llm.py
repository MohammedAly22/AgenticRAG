"""
An abstract base class defining the interface for all Large Language Model (LLM) wrappers.

This class enforces a consistent `generate` method across all LLM implementations,
allowing them to be used interchangeably within the application.

Subclasses must implement the `generate` method to support text generation.

Methods:
    generate(prompt: str, stream: bool = False) -> str:
        Abstract method to generate a response from the LLM based on a given prompt.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model (LLM) implementations.
    """

    @abstractmethod
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate a response from the LLM given a prompt.

        Args:
            prompt (str): The input text prompt to generate a response for.
            stream (bool): Whether to stream the output (e.g., for real-time UIs). Defaults to False.

        Returns:
            str: The generated text response.
        """
        
        pass
