"""
A wrapper class for initializing and accessing Hugging Face embedding models
using LangChain's HuggingFaceEmbeddings.

This class provides a simplified interface for loading sentence-transformer models
for tasks such as semantic search, vector indexing, and document similarity.

Attributes:
    - embeddings (HuggingFaceEmbeddings): An instance of LangChain's HuggingFaceEmbeddings.
"""

from langchain_huggingface.embeddings import HuggingFaceEmbeddings


class HFEmbedding:
    """
    A utility class to initialize and access Hugging Face embedding models.

    Args:
        - model_name (str): The name of the Hugging Face embedding model to use.
                          Defaults to "sentence-transformers/all-mpnet-base-v2".
    """

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the Hugging Face embedding model with the specified model name.
        """

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
        )

    def get_embeddings(self):
        """
        Returns the initialized HuggingFaceEmbeddings instance.

        Returns:
            - HuggingFaceEmbeddings: The embedding model ready for use.
        """

        return self.embeddings
