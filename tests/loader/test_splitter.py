import unittest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from deepsearcher.loader.splitter import Chunk, split_docs_to_chunks, _sentence_window_split


class TestSplitter(unittest.TestCase):
    """Tests for the splitter module."""
    
    def test_chunk_init(self):
        """Test initialization of Chunk class."""
        # Test with minimal parameters
        chunk = Chunk(text="Test text", reference="test_ref")
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.reference, "test_ref")
        self.assertEqual(chunk.metadata, {})
        self.assertIsNone(chunk.embedding)
        
        # Test with all parameters
        metadata = {"key": "value"}
        embedding = [0.1, 0.2, 0.3]
        chunk = Chunk(text="Test text", reference="test_ref", metadata=metadata, embedding=embedding)
        self.assertEqual(chunk.text, "Test text")
        self.assertEqual(chunk.reference, "test_ref")
        self.assertEqual(chunk.metadata, metadata)
        self.assertEqual(chunk.embedding, embedding)
    
    def test_sentence_window_split(self):
        """Test _sentence_window_split function."""
        # Create a test document
        original_text = "This is a test document. It has multiple sentences. This is for testing the splitter."
        original_doc = Document(page_content=original_text, metadata={"reference": "test_doc"})
        
        # Create split documents
        split_docs = [
            Document(page_content="This is a test document.", metadata={"reference": "test_doc"}),
            Document(page_content="It has multiple sentences.", metadata={"reference": "test_doc"}),
            Document(page_content="This is for testing the splitter.", metadata={"reference": "test_doc"})
        ]
        
        # Test with default offset
        chunks = _sentence_window_split(split_docs, original_doc)
        
        # Verify the results
        self.assertEqual(len(chunks), 3)
        for i, chunk in enumerate(chunks):
            self.assertEqual(chunk.text, split_docs[i].page_content)
            self.assertEqual(chunk.reference, "test_doc")
            self.assertIn("wider_text", chunk.metadata)
            # The wider text should contain the original text since our test document is short
            self.assertEqual(chunk.metadata["wider_text"], original_text)
        
        # Test with smaller offset
        chunks = _sentence_window_split(split_docs, original_doc, offset=10)
        
        # Verify the results with smaller context windows
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            # With smaller offset, wider_text should be shorter than the full original text
            self.assertLessEqual(len(chunk.metadata["wider_text"]), len(original_text))
    
    def test_sentence_window_split_repeated_text(self):
        """Test that repeated text gets the context window of its own position."""
        original_text = "x" + "." * 20 + "y" + "-" * 20 + "x"
        original_doc = Document(page_content=original_text, metadata={"reference": "test_doc"})
        split_docs = [
            Document(page_content="x", metadata={"reference": "test_doc"}),
            Document(page_content="y", metadata={"reference": "test_doc"}),
            Document(page_content="x", metadata={"reference": "test_doc"}),
        ]

        chunks = _sentence_window_split(split_docs, original_doc, offset=3)

        self.assertEqual(chunks[0].metadata["wider_text"], "x...")
        self.assertEqual(chunks[1].metadata["wider_text"], "...y---")
        self.assertEqual(chunks[2].metadata["wider_text"], "---x")

    def test_sentence_window_split_text_inside_previous_chunk(self):
        """Test that a chunk whose text also occurs inside the previous chunk is not placed there."""
        original_doc = Document(page_content="XABCABC", metadata={"reference": "test_doc"})
        split_docs = [
            Document(page_content="XABC", metadata={"reference": "test_doc"}),
            Document(page_content="ABC", metadata={"reference": "test_doc"}),
        ]

        chunks = _sentence_window_split(split_docs, original_doc, offset=1)

        self.assertEqual(chunks[0].metadata["wider_text"], "XABCA")
        self.assertEqual(chunks[1].metadata["wider_text"], "CABC")

    def test_sentence_window_split_matches_splitter_positions(self):
        """Test that chunk positions match the splitter's own start indexes."""
        for original_text, chunk_size, chunk_overlap in [
            ("ab ab ab ab ab", 5, 0),
            ("x ab ab", 6, 1),
        ]:
            with self.subTest(original_text=original_text, chunk_overlap=chunk_overlap):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
                )
                split_docs = splitter.split_documents([Document(page_content=original_text)])
                expected = []
                for d in split_docs:
                    start = d.metadata["start_index"]
                    end = start + len(d.page_content)
                    expected.append(original_text[max(0, start - 1) : end + 1])

                chunks = _sentence_window_split(
                    split_docs,
                    Document(page_content=original_text),
                    offset=1,
                    chunk_overlap=chunk_overlap,
                )

                self.assertEqual([c.metadata["wider_text"] for c in chunks], expected)

    def test_sentence_window_split_text_not_in_original(self):
        """Test that a chunk missing from the original text falls back to its own text."""
        original_doc = Document(page_content="alpha beta", metadata={"reference": "test_doc"})
        split_docs = [
            Document(page_content="alpha", metadata={"reference": "test_doc"}),
            Document(page_content="gamma", metadata={"reference": "test_doc"}),
            Document(page_content="beta", metadata={"reference": "test_doc"}),
        ]

        chunks = _sentence_window_split(split_docs, original_doc, offset=2)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["wider_text"], "alpha b")
        self.assertEqual(chunks[1].metadata["wider_text"], "gamma")
        self.assertEqual(chunks[2].metadata["wider_text"], "a beta")

    def test_split_docs_to_chunks(self):
        """Test split_docs_to_chunks function."""
        # Create test documents
        docs = [
            Document(
                page_content="This is document one. It has some content for testing.",
                metadata={"reference": "doc1"}
            ),
            Document(
                page_content="This is document two. It also has content for testing purposes.",
                metadata={"reference": "doc2"}
            )
        ]
        
        # Test with default parameters
        chunks = split_docs_to_chunks(docs)
        
        # Verify the results
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIn(chunk.reference, ["doc1", "doc2"])
            self.assertIn("wider_text", chunk.metadata)
        
        # Test with custom chunk size and overlap
        chunks = split_docs_to_chunks(docs, chunk_size=10, chunk_overlap=2)
        
        # With small chunk size, we should get more chunks
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            self.assertIsInstance(chunk, Chunk)
            self.assertIn(chunk.reference, ["doc1", "doc2"])
            self.assertIn("wider_text", chunk.metadata)


if __name__ == "__main__":
    unittest.main() 