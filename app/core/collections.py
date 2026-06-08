### helper functions ###
import json
from pathlib import Path
from fastapi import HTTPException

from app.core.config import settings

def get_stored_collections() -> list[str]:
    ''' Read the collections.json file and return the list of collection names. If the file doesn't exist, return an empty list. '''
    if Path(settings.collections_file).exists():
        with open(settings.collections_file, "r") as f:
            return json.load(f)["collections"]
    else:
        return []

def save_collection(collection_name: str) -> None:
    ''' Save the collection name to the collections.json file. '''
    existing_collections = get_stored_collections()

    if len(existing_collections) >= 3:
        raise HTTPException(status_code=400, detail="You have reached the maximum number of collections. Please delete a collection before adding a new one.")
    else:
        with open(settings.collections_file, "w") as f:
            json.dump({"collections": existing_collections + [collection_name]}, f)
