"""
A concrete implementation of the BaseLLM interface using Cohere's chat model via LangChain.

This class wraps the Cohere LLM (`command` family models) and provides a unified `generate`
method that supports both standard and streaming generation.

Attributes:
    - chat_model (ChatCohere): An instance of LangChain's ChatCohere for interacting with the Cohere API.

Methods:
    - get_llm(): Returns the underlying ChatCohere instance.
    - generate(prompt: str, stream: bool): Generates a text response from the Cohere LLM.
"""

from typing import Union, Iterator
from langchain_cohere.chat_models import ChatCohere

# local
from src.llms.base_llm import BaseLLM


class CohereLLM(BaseLLM):
    """
    Cohere LLM implementation of the BaseLLM interface using LangChain's ChatCohere.

    Args:
        - cohere_api_key (str): The API key for authenticating with Cohere.
        - model_name (str): The Cohere model name to use. Defaults to "command-a-03-2025".
    """

    def __init__(self, cohere_api_key: str, model_name: str = "command-a-03-2025"):
        super().__init__()
        self.chat_model = ChatCohere(model=model_name, cohere_api_key=cohere_api_key)

    def get_llm(self):
        """
        Returns the initialized ChatCohere instance.

        Returns:
           - ChatCohere: The Cohere chat model instance.
        """

        return self.chat_model

    def generate(self, prompt: str, stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Generate a response from the Cohere LLM based on the given prompt.

        Args:
            - prompt (str): The text prompt to send to the LLM.
            - stream (bool): Whether to stream the output. Defaults to False.

        Returns:
            - Union[str, Iterator[str]]: The generated response as a string,
            or an iterator of streamed tokens if `stream=True`.
        """

        if stream:
            return self.chat_model.stream(prompt)
        else:
            return self.chat_model.invoke(prompt).content
