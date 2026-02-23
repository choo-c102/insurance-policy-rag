from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

''' Split list of documents into smaller chunks for embedding '''
def split_into_chunks(
    documents: list[Document],
    chunk_size: int = 1000, 
    chunk_overlap: int = 200, 
    add_start_index: bool = True
) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        add_start_index=add_start_index
        )
    return text_splitter.split_documents(documents)