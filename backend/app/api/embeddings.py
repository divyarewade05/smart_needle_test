"""
Embeddings API
--------------
Routes for managing reference embeddings.

POST /embeddings/rebuild   → Smart rebuild (only dirty persons)
POST /embeddings/rebuild?force=true → Full rebuild
GET  /embeddings           → List all stored identities
GET  /embeddings/{name}    → Get one identity's metadata
DELETE /embeddings/{name}  → Remove an identity from store
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from app.core.config import get_settings, get_engine, get_store
from app.ml.reference_builder import build_reference_embeddings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post("/rebuild")
def rebuild_embeddings(force: bool = Query(False, description="Force full rebuild")):
    """
    Rebuild reference embeddings.
    - By default: only recomputes changed/new/deleted persons (smart).
    - With force=true: recomputes all persons from scratch.
    """
    settings = get_settings()
    engine = get_engine()
    store = get_store()

    log_events = []

    def capture(event, payload):
        log_events.append({"event": event, "data": payload})

    summary = build_reference_embeddings(
        reference_path=settings.reference_path,
        store=store,
        engine=engine,
        force_rebuild=force,
        progress_callback=capture,
    )

    return {
        "status": "ok",
        "force": force,
        "summary": summary,
        "log": log_events,
    }


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