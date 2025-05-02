"""
Utility class for splitting documents into smaller chunks using LangChain's text splitters.

This class currently uses `RecursiveCharacterTextSplitter` with large chunk sizes
to maintain context and reduce the number of splits.
"""

from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:
    """
    A wrapper class for document splitting using RecursiveCharacterTextSplitter.

    Attributes:
        - text_splitter (RecursiveCharacterTextSplitter): Instance configured for chunking documents.
    """

    def __init__(self):
        """
        Initialize the TextSplitter with a large chunk size and some overlap to retain context.
        Uses common text separators for recursive splitting.
        """

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,  # Large chunk size to reduce number of chunks
            chunk_overlap=800,  # Maintain some overlap for context
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, docs: List[Document]):
        """
        Split a list of LangChain Document objects into smaller chunks.

        Args:
            - docs (List[Document]): A list of documents to be split.

        Returns:
            - List[Document]: A list of document chunks after splitting.
        """

        text_splits = self.text_splitter.split_documents(docs)

        return text_splits
