'''
Accept a PDF file from the user
Save it temporarily to disk
Run it through load_documents → split_into_chunks → create_vector_store
Return a confirmation message
'''

from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from pathlib import Path

from app.rag.loader import load_documents
from app.rag.chunker import split_into_chunks
from app.rag.vectorstore import create_vector_store
from app.core.collections import save_collection

router = APIRouter()

@router.post("/upload")
async def upload_process_policy(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):  # Validate uploaded file is a PDF
        raise HTTPException(status_code=400, detail="File must be a PDF")

    Path("temp").mkdir(exist_ok=True)

    temp_path = Path(f"temp/{file.filename}")
    collection_name = f"{file.filename.split('.')[0]}_collection"

    with open(str(temp_path), "wb") as temp_file: # Save file temporarily to temp/{file.filename}
        shutil.copyfileobj(file.file, temp_file)

    try:
        documents = load_documents(str(temp_path)) # Load documents
        chunks = split_into_chunks(documents) # Split documents into chunks
        create_vector_store(
            chunks, 
            collection_name=collection_name
        ) # Create vector store
        save_collection(collection_name)

    finally:
        if temp_path.exists():
            temp_path.unlink()
        if Path("temp").exists():
            shutil.rmtree("temp")

    return {
        "message": f"File uploaded successfully",
        "collection_name": collection_name,
        "pages_loaded": len(documents),
        "chunks_created": len(chunks)
    }