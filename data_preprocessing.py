

### 3. src/data_preprocessing.py
'''
This module handles loading the pickle file, downloading necessary NLTK data, and chunking the document text.

'''

# src/data_preprocessing.py

import pickle
import nltk
from nltk.tokenize import sent_tokenize

def download_nltk_data():
    """Download required NLTK tokenizer data."""
    nltk.download('punkt')
    nltk.download('punkt_tab')

def load_pickle(file_path):
    """Load and return the pickle file data."""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def extract_text_from_doc(doc):
    """
    Extract the text from a document object.
    Assumes the document has a nested structure: doc.text_resource.text
    """
    return doc.text_resource.text

def chunk_markdown_text(md_text, max_words=300):
    """
    Chunk the markdown text into smaller pieces based on sentence boundaries.
    Preserves markdown formatting.
    
    Args:
        md_text (str): The markdown text.
        max_words (int): Maximum number of words per chunk.
        
    Returns:
        List[str]: A list of text chunks.
    """
    sentences = sent_tokenize(md_text)
    chunks = []
    current_chunk = ""
    current_words = 0
    for sentence in sentences:
        words = sentence.split()
        if current_words + len(words) <= max_words:
            current_chunk += " " + sentence
            current_words += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_words = len(words)
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
