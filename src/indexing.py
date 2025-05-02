"""
Module for managing a Chroma vector store index using LangChain.
"""

from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


class Index:
    """
    Wrapper class for creating and interacting with a Chroma vector store.

    Attributes:
        - vector_store (Chroma): An instance of the Chroma vector store.
    """

    def __init__(self, embeddings):
        """
        Initialize the Chroma vector store with a specified embedding function.

        Args:
            - embeddings: A LangChain-compatible embedding function.
        """

        self.vector_store = Chroma(
            collection_name="index-ai-report-2025",
            embedding_function=embeddings,
            persist_directory="./chroma-langchain-db",
        )

    def add_documents(self, docs: List[Document]):
        """
        Add a list of documents to the vector store.

        Args:
            - docs (List[Document]): List of LangChain Document objects to be indexed.
        """

        _ = self.vector_store.add_documents(documents=docs)

    def get_vector_store(self):
        """
        Retrieve the internal Chroma vector store instance.

        Returns:
            - Chroma: The Chroma vector store object.
        """

        return self.vector_store
