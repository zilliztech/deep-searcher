import os
from abc import ABC
from typing import List

from langchain_core.documents import Document


class BaseLoader(ABC):
    """
    Abstract base class for file loaders.

    This class defines the interface for loading documents from files and directories.
    All specific file loaders should inherit from this class and implement the required methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the loader with optional keyword arguments.

        Args:
            **kwargs: Optional keyword arguments for specific loader implementations.
        """
        pass

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file and convert it to Document objects.

        Args:
            file_path: Path to the file to be loaded.

        Returns:
            A list of Document objects containing the text and metadata.

        Note:
            Return a list of Document objects which contain the text and metadata.
            In the metadata, it's recommended to include the reference to the file.
            e.g. return [Document(page_content=..., metadata={"reference": file_path})]
        """
        pass

    def load_directory(self, directory: str) -> List[Document]:
        """
        Load all supported files from a directory and its subdirectories recursively.

        Args:
            directory: Path to the directory containing files to be loaded.

        Returns:
            A list of Document objects from all supported files in the directory and subdirectories.
        """
        documents = []
        for root, _, files in os.walk(directory):
            for file in files:
                for suffix in self.supported_file_types:
                    if file.endswith("." + suffix.lstrip(".")):
                        full_path = os.path.join(root, file)
                        documents.extend(self.load_file(full_path))
                        break
        return documents

    @property
    def supported_file_types(self) -> List[str]:
        """
        Get the list of file extensions supported by this loader.

        Returns:
            A list of supported file extensions (without the dot).
        """
        pass
