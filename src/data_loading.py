"""
Module for loading PDF documents using LangChain's PyMuPDFLoader.
"""

from langchain_community.document_loaders import PyMuPDFLoader


class DataLoader:
    """
    A utility class for loading PDF documents using PyMuPDFLoader.

    Attributes:
        - loader (PyMuPDFLoader): The document loader initialized with the given file path.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with the path to a PDF file.

        Args:
            - file_path (str): Path to the PDF document to be loaded.
        """

        self.loader = PyMuPDFLoader(file_path=file_path)

    def get_docs(self):
        """
        Load and return the documents from the specified PDF.

        Returns:
            - List[Document]: A list of LangChain Document objects extracted from the PDF.
        """

        docs = self.loader.load()

        return docs
