"""
A factory class for creating embedding model instances from different providers.

This class abstracts the instantiation of embedding models (e.g., Cohere, HuggingFace)
and allows easy switching between providers by passing a provider name and relevant arguments.

Supported providers:
- "cohere": Uses `CohereEmbedding` (requires `cohere_api_key`)
- "huggingface": Uses `HFEmbedding`

Raises:
    - AssertionError: If `cohere_api_key` is missing when using the "cohere" provider.
    - ValueError: If an unsupported provider is specified.
"""


from src.embeddings.cohere_embeddings import CohereEmbedding
from src.embeddings.hf_embeddings import HFEmbedding


class EmbeddingsFactory:
    """
    A factory class to create embedding model instances from specified providers.
    """

    @staticmethod
    def create_embeddings(provider: str, **kwargs):
        """
        Creates and returns an instance of an embedding model based on the provider.

        Args:
            - provider (str): The name of the embedding model provider ("cohere" or "huggingface").
            - **kwargs: Additional keyword arguments required for model initialization.
                - For "cohere": must include `cohere_api_key`
                - For "huggingface": optionally include `model_name`

        Returns:
            - An instance of `CohereEmbedding` or `HFEmbedding`.

        Raises:
            - AssertionError: If required parameters are missing for a provider.
            - ValueError: If the provider name is unsupported.
        """
        
        if provider.lower() == "cohere":
            assert "cohere_api_key" in kwargs, "Please, enter pass `cohere_api_key` argument"
            return CohereEmbedding(**kwargs)

        elif provider.lower() == "huggingface":
            return HFEmbedding(**kwargs)

        else:
            raise ValueError(f"Unsupported embeddings model provider: {provider}")
