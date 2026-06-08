from fastapi import FastAPI
from app.api.routes import upload, query
from app.core.collections import get_stored_collections

app = FastAPI()

app.include_router(upload.router, prefix="/api")
app.include_router(query.router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Policy RAG API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/available_policies")
def get_available_policies():
    return {"available_policies": get_stored_collections()} #Return the list of available policies