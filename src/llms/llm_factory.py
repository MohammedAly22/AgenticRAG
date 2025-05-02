"""
Factory class for creating LLM (Large Language Model) instances.

This utility class simplifies the initialization of LLMs by selecting the appropriate
implementation based on a provided `llm_type`. Currently, only Cohere is supported.

Methods:
    - create_llm(llm_type: str, **kwargs): Returns an instance of a subclass of BaseLLM
    - based on the given type and configuration.
"""


from src.llms.cohere_llm import CohereLLM


class LLMFactory:
    """
    Factory class for creating LLM instances based on a specified provider type.
    """

    @staticmethod
    def create_llm(llm_type: str, **kwargs):
        """
        Create and return an instance of an LLM based on the specified type.

        Args:
            - llm_type (str): The type/provider of the LLM (e.g., "cohere").
            - **kwargs: Additional keyword arguments required for initializing the LLM,
                      such as API keys or model names.

        Returns:
            - BaseLLM: An instance of a class implementing the BaseLLM interface.

        Raises:
            - AssertionError: If required arguments are missing (e.g., `cohere_api_key` for Cohere).
            - ValueError: If the specified `llm_type` is not supported.
        """

        if llm_type.lower() == "cohere":
            assert (
                "cohere_api_key" in kwargs
            ), "Please, enter pass `cohere_api_key` argument"
            return CohereLLM(**kwargs)

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
