from fastapi import FastAPI

app = FastAPI(title = "Insurance Policy RAG API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}