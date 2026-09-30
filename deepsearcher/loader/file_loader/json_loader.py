import json
from typing import List

from langchain_core.documents import Document

from deepsearcher.loader.file_loader.base import BaseLoader


class JsonFileLoader(BaseLoader):
    """
    Loader for JSON and JSONL files.

    This loader handles JSON and JSONL files, extracting text content from a specified key
    and converting each entry into Document objects for further processing.
    """

    def __init__(self, text_key: str):
        """
        Initialize the JsonFileLoader.

        Args:
            text_key: The key in the JSON data that contains the text content to be extracted.
        """
        self.text_key = text_key

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a JSON or JSONL file and convert it to Document objects.

        Args:
            file_path: Path to the JSON or JSONL file to be loaded.

        Returns:
            A list of Document objects, one for each entry in the JSON/JSONL file.
        """
        if file_path.endswith(".jsonl"):
            data_list: list[dict] = self._read_jsonl_file(file_path)
        else:
            data_list: list[dict] = self._read_json_file(file_path)
        documents = []
        for data_dict in data_list:
            page_content = data_dict.pop(self.text_key)
            data_dict.update({"reference": file_path})
            document = Document(page_content=page_content, metadata=data_dict)
            documents.append(document)
        return documents

    def _read_json_file(self, file_path: str) -> list[dict]:
        """
        Read and parse a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            A list of dictionaries parsed from the JSON file.

        Raises:
            ValueError: If the JSON file does not contain a list of dictionaries.
        """
        json_data = json.load(open(file_path))
        if not isinstance(json_data, list):
            raise ValueError("JSON file must contain a list of dictionaries.")
        return json_data

    def _read_jsonl_file(self, file_path: str) -> List[dict]:
        """
        Read and parse a JSONL file (JSON Lines format).

        Args:
            file_path: Path to the JSONL file.

        Returns:
            A list of dictionaries parsed from the JSONL file.
        """
        data_list = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    json_data = json.loads(line)
                    data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Failed to decode line: {line}")
        return data_list

    @property
    def supported_file_types(self) -> List[str]:
        """
        Get the list of file extensions supported by this loader.

        Returns:
            A list of supported file extensions: ["json", "jsonl"].
        """
        return ["json", "jsonl"]
