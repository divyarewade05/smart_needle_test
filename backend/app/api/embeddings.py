"""
Embeddings API
--------------
Routes for managing reference embeddings.
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.core.config import get_settings, get_engine, get_store
from app.ml.reference_builder import build_reference_embeddings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post("/rebuild")
def rebuild_embeddings(force: bool = Query(False, description="Force full rebuild")):
    """
    Streamed rebuild of reference embeddings.
    Prevents timeouts and allows real-time UI logging.
    """
    settings = get_settings()
    engine = get_engine()
    store = get_store()

    def stream():
        # build_reference_embeddings is now a generator
        for event in build_reference_embeddings(
            reference_path=settings.reference_path,
            store=store,
            engine=engine,
            force_rebuild=force,
        ):
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@router.get("")
def list_identities():
    """List all stored identities with metadata (no raw embeddings)."""
    store = get_store()
    identities = store.list_identities()
    return {
        "total": len(identities),
        "identities": identities,
    }


@router.get("/{name}")
def get_identity(name: str):
    """Get metadata for a single stored identity."""
    store = get_store()
    entry = store.get_person(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    return {
        "name": name,
        "sample_count": entry["sample_count"],
        "updated_at": entry["updated_at"],
    }


@router.delete("/{name}")
def delete_identity(name: str):
    """Remove an identity from the store and save."""
    store = get_store()
    entry = store.get_person(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Identity '{name}' not found.")
    store.remove(name)
    store.save()
    return {"status": "deleted", "name": name}