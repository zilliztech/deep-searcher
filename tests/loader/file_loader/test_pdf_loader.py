import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from deepsearcher.loader.file_loader import PDFLoader


class TestPDFLoader(unittest.TestCase):
    """Tests for the PDFLoader class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a text file for testing
        self.text_file_path = os.path.join(self.temp_dir.name, "test.txt")
        with open(self.text_file_path, "w", encoding="utf-8") as f:
            f.write("This is a test text file.")
        
        # Create a markdown file for testing
        self.md_file_path = os.path.join(self.temp_dir.name, "test.md")
        with open(self.md_file_path, "w", encoding="utf-8") as f:
            f.write("# Test Markdown\nThis is a test markdown file.")
        
        # PDF file path (will be mocked)
        self.pdf_file_path = os.path.join(self.temp_dir.name, "test.pdf")
        
        # Create the loader
        self.loader = PDFLoader()
    
    def tearDown(self):
        """Clean up the test environment."""
        self.temp_dir.cleanup()
    
    def test_supported_file_types(self):
        """Test the supported_file_types property."""
        file_types = self.loader.supported_file_types
        self.assertIsInstance(file_types, list)
        self.assertIn("pdf", file_types)
        self.assertIn("md", file_types)
        self.assertIn("txt", file_types)
    
    def test_load_text_file(self):
        """Test loading a text file."""
        documents = self.loader.load_file(self.text_file_path)
        
        # Check that we got one document
        self.assertEqual(len(documents), 1)
        
        # Check the document content
        document = documents[0]
        self.assertEqual(document.page_content, "This is a test text file.")
        
        # Check the metadata
        self.assertEqual(document.metadata["reference"], self.text_file_path)
    
    def test_load_markdown_file(self):
        """Test loading a markdown file."""
        documents = self.loader.load_file(self.md_file_path)
        
        # Check that we got one document
        self.assertEqual(len(documents), 1)
        
        # Check the document content
        document = documents[0]
        self.assertEqual(document.page_content, "# Test Markdown\nThis is a test markdown file.")
        
        # Check the metadata
        self.assertEqual(document.metadata["reference"], self.md_file_path)
    
    @patch("pdfplumber.open")
    def test_load_pdf_file(self, mock_pdf_open):
        """Test loading a PDF file."""
        # Set up mock PDF pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        # Set up mock PDF file
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdf.__enter__.return_value = mock_pdf
        mock_pdf.__exit__.return_value = None
        
        # Configure the mock to return our mock PDF
        mock_pdf_open.return_value = mock_pdf
        
        # Create a dummy PDF file
        with open(self.pdf_file_path, "w") as f:
            f.write("dummy pdf content")
        
        # Load the PDF file
        documents = self.loader.load_file(self.pdf_file_path)
        
        # Verify pdfplumber.open was called
        mock_pdf_open.assert_called_once_with(self.pdf_file_path)
        
        # Check that we got one document
        self.assertEqual(len(documents), 1)
        
        # Check the document content
        document = documents[0]
        self.assertEqual(document.page_content, "Page 1 content\n\nPage 2 content")
        
        # Check the metadata
        self.assertEqual(document.metadata["reference"], self.pdf_file_path)
    
    def test_load_directory(self):
        """Test loading a directory with mixed file types."""
        # Create the loader
        loader = PDFLoader()
        
        # Mock the load_file method to track calls
        original_load_file = loader.load_file
        calls = []
        
        def mock_load_file(file_path):
            calls.append(file_path)
            return original_load_file(file_path)
        
        loader.load_file = mock_load_file
        
        # Load the directory
        documents = loader.load_directory(self.temp_dir.name)
        
        # Check that we processed both text and markdown files
        self.assertEqual(len(calls), 2)  # text and markdown files
        self.assertIn(self.text_file_path, calls)
        self.assertIn(self.md_file_path, calls)
        
        # Check that we got two documents
        self.assertEqual(len(documents), 2)

    def test_load_directory_ignores_names_that_only_end_with_extension_letters(self):
        """A file like build.cmd ends with "md" but is not a .md file."""
        cmd_path = os.path.join(self.temp_dir.name, "build.cmd")
        with open(cmd_path, "w", encoding="utf-8") as f:
            f.write("@echo off")
        documents = PDFLoader().load_directory(self.temp_dir.name)
        self.assertEqual(len(documents), 2)
        self.assertNotIn(cmd_path, [d.metadata["reference"] for d in documents])


if __name__ == "__main__":
    unittest.main() 