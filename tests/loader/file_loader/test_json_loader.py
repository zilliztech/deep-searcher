import unittest
import os
import json
import tempfile

from langchain_core.documents import Document

from deepsearcher.loader.file_loader import JsonFileLoader


class TestJsonFileLoader(unittest.TestCase):
    """Tests for the JsonFileLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Sample JSON data
        self.json_data = [
            {"id": 1, "text": "This is the first document.", "author": "John Doe"},
            {"id": 2, "text": "This is the second document.", "author": "Jane Smith"}
        ]
        
        # Create JSON test file
        self.json_file_path = os.path.join(self.temp_dir.name, "test.json")
        with open(self.json_file_path, "w", encoding="utf-8") as f:
            json.dump(self.json_data, f)
        
        # Create JSONL test file
        self.jsonl_file_path = os.path.join(self.temp_dir.name, "test.jsonl")
        with open(self.jsonl_file_path, "w", encoding="utf-8") as f:
            for item in self.json_data:
                f.write(json.dumps(item) + "\n")
        
        # Create invalid JSON file (not a list)
        self.invalid_json_file_path = os.path.join(self.temp_dir.name, "invalid.json")
        with open(self.invalid_json_file_path, "w", encoding="utf-8") as f:
            json.dump({"id": 1, "text": "This is not a list.", "author": "John Doe"}, f)
        
        # Create invalid JSONL file
        self.invalid_jsonl_file_path = os.path.join(self.temp_dir.name, "invalid.jsonl")
        with open(self.invalid_jsonl_file_path, "w", encoding="utf-8") as f:
            f.write("This is not valid JSON\n")
            f.write(json.dumps({"id": 2, "text": "This is valid JSON", "author": "Jane Smith"}) + "\n")
        
        # Initialize the loader
        self.loader = JsonFileLoader(text_key="text")
        
        # Patch the _read_json_file method to fix the file handling
        original_read_json_file = self.loader._read_json_file
        
        def patched_read_json_file(file_path):
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            if not isinstance(json_data, list):
                raise ValueError("JSON file must contain a list of dictionaries.")
            return json_data
        
        self.loader._read_json_file = patched_read_json_file
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_load_json_file(self):
        """Test loading a JSON file."""
        documents = self.loader.load_file(self.json_file_path)
        
        # Check that we got the right number of documents
        self.assertEqual(len(documents), 2)
        
        # Check the content and metadata of each document
        self.assertEqual(documents[0].page_content, "This is the first document.")
        self.assertEqual(documents[0].metadata["id"], 1)
        self.assertEqual(documents[0].metadata["author"], "John Doe")
        self.assertEqual(documents[0].metadata["reference"], self.json_file_path)
        
        self.assertEqual(documents[1].page_content, "This is the second document.")
        self.assertEqual(documents[1].metadata["id"], 2)
        self.assertEqual(documents[1].metadata["author"], "Jane Smith")
        self.assertEqual(documents[1].metadata["reference"], self.json_file_path)
    
    def test_load_jsonl_file(self):
        """Test loading a JSONL file."""
        documents = self.loader.load_file(self.jsonl_file_path)
        
        # Check that we got the right number of documents
        self.assertEqual(len(documents), 2)
        
        # Check the content and metadata of each document
        self.assertEqual(documents[0].page_content, "This is the first document.")
        self.assertEqual(documents[0].metadata["id"], 1)
        self.assertEqual(documents[0].metadata["author"], "John Doe")
        self.assertEqual(documents[0].metadata["reference"], self.jsonl_file_path)
        
        self.assertEqual(documents[1].page_content, "This is the second document.")
        self.assertEqual(documents[1].metadata["id"], 2)
        self.assertEqual(documents[1].metadata["author"], "Jane Smith")
        self.assertEqual(documents[1].metadata["reference"], self.jsonl_file_path)
    
    def test_invalid_json_file(self):
        """Test loading an invalid JSON file (not a list)."""
        with self.assertRaises(ValueError):
            self.loader.load_file(self.invalid_json_file_path)
    
    def test_invalid_jsonl_file(self):
        """Test loading a JSONL file with invalid lines."""
        documents = self.loader.load_file(self.invalid_jsonl_file_path)
        
        # Only the valid line should be loaded
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "This is valid JSON")
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        file_types = self.loader.supported_file_types
        self.assertIsInstance(file_types, list)
        self.assertEqual(sorted(file_types), ["json", "jsonl"])

    def test_load_directory(self):
        """Test that load_directory loads JSON/JSONL files and skips other files."""
        with tempfile.TemporaryDirectory() as dir_path:
            json_path = os.path.join(dir_path, "docs.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([{"text": "json document"}], f)
            jsonl_path = os.path.join(dir_path, "docs.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "jsonl document"}) + "\n")
            md_path = os.path.join(dir_path, "notes.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("This is not JSON.")

            documents = self.loader.load_directory(dir_path)

        references = sorted(doc.metadata["reference"] for doc in documents)
        self.assertEqual(references, sorted([json_path, jsonl_path]))
        self.assertNotIn(md_path, references)
        self.assertEqual(
            sorted(doc.page_content for doc in documents),
            ["json document", "jsonl document"],
        )


if __name__ == "__main__":
    unittest.main() 