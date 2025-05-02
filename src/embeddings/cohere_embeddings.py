"""
A wrapper class for initializing and accessing Cohere embedding models using LangChain's CohereEmbeddings.

This class simplifies the process of setting up and retrieving the embedding model for downstream tasks
such as document indexing or semantic search.

Attributes:
    - embeddings (CohereEmbeddings): An instance of the LangChain CohereEmbeddings initialized with the specified model and API key.
"""

from langchain_cohere.embeddings import CohereEmbeddings


class CohereEmbedding:
    """
    A utility class to initialize and access Cohere embedding models.

    Args:
        - cohere_api_key (str): Your Cohere API key.
        - model_name (str): The name of the Cohere embedding model to use. Defaults to "embed-v4.0".
    """
    
    def __init__(self, cohere_api_key: str, model_name: str = "embed-v4.0"):
        """
        Initialize the Cohere embedding model with the specified API key and model name.
        """
        self.embeddings = CohereEmbeddings(
            model=model_name, cohere_api_key=cohere_api_key
        )

    def get_embeddings(self):
        """
        Returns the initialized CohereEmbeddings instance.

        Returns:
            - CohereEmbeddings: The embedding model ready for use.
        """
        return self.embeddings
