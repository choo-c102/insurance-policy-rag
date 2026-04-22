import os
from langchain_community.document_loaders import PyMuPDFLoader 

def load_documents(file_path: str):
    '''
    Load PDF document from a given path 
    Returns a list of documents (each document is a page in the PDF)
    '''

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No file found at path: {file_path}")
    
    if not file_path.lower().endswith('.pdf'):
        raise ValueError(f"Expected a PDF file, got: {file_path}")
    
    try:
        loader = PyMuPDFLoader(file_path, extract_tables_settings={"enabled": True})
        return loader.load() 
    
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the document: {e}")

    
