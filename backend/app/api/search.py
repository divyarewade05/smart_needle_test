# """
# Search API
# ----------
# Routes for searching by identity across image collections.

# POST /search/person        → Search for one person across all test images
# POST /search/index-all     → Index all persons across all test images
# GET  /search/persons       → List all persons found in test images (from last index)
# """

# import os
# from fastapi import APIRouter, HTTPException, Query
# from pydantic import BaseModel
# from typing import Optional

# from app.core.config import get_settings, get_store, get_searcher

# router = APIRouter(prefix="/search", tags=["Search"])


# class PersonSearchRequest(BaseModel):
#     name: str
#     image_folder: Optional[str] = None
#     save_annotated: bool = True


# @router.post("/person")
# def search_person(request: PersonSearchRequest):
#     """
#     Search for a specific person across an image collection.

#     Returns all images where this person's face is detected,
#     regardless of whether the image contains other faces.
#     """
#     settings = get_settings()
#     store = get_store()
#     searcher = get_searcher()

#     reference_embeddings = store.get_all_embeddings()
#     if not reference_embeddings:
#         raise HTTPException(
#             status_code=422,
#             detail="No reference embeddings. Run /embeddings/rebuild first."
#         )

#     if request.name not in reference_embeddings:
#         available = list(reference_embeddings.keys())
#         raise HTTPException(
#             status_code=404,
#             detail=f"Identity '{request.name}' not found. Available: {available}"
#         )

#     image_folder = request.image_folder or settings.test_images_path

#     if not os.path.isdir(image_folder):
#         raise HTTPException(
#             status_code=404,
#             detail=f"Image folder not found: {image_folder}"
#         )

#     output_folder = os.path.join(
#         settings.output_path,
#         "search",
#         request.name.replace(" ", "_"),
#     )

#     log_events = []
#     def capture(event, payload):
#         log_events.append({"event": event, "data": payload})

#     result = searcher.search_person(
#         query_name=request.name,
#         image_folder=image_folder,
#         reference_embeddings=reference_embeddings,
#         output_folder=output_folder,
#         save_annotated=request.save_annotated,
#         progress_callback=capture,
#     )

#     return {
#         **result,
#         "output_folder": output_folder if request.save_annotated else None,
#         "log": log_events,
#     }


# @router.post("/index-all")
# def index_all(
#     image_folder: Optional[str] = Query(None),
#     save_annotated: bool = Query(True),
# ):
#     """
#     Process ALL images and build a full index of who appears where.
#     Returns a person → [image list] mapping.
#     """
#     settings = get_settings()
#     store = get_store()
#     searcher = get_searcher()

#     reference_embeddings = store.get_all_embeddings()
#     if not reference_embeddings:
#         raise HTTPException(
#             status_code=422,
#             detail="No reference embeddings. Run /embeddings/rebuild first."
#         )

#     target_folder = image_folder or settings.test_images_path
#     if not os.path.isdir(target_folder):
#         raise HTTPException(
#             status_code=404,
#             detail=f"Image folder not found: {target_folder}"
#         )

#     output_folder = os.path.join(settings.output_path, "index")

#     log_events = []
#     def capture(event, payload):
#         log_events.append({"event": event, "data": payload})

#     result = searcher.search_all_persons(
#         image_folder=target_folder,
#         reference_embeddings=reference_embeddings,
#         output_folder=output_folder,
#         save_annotated=save_annotated,
#         progress_callback=capture,
#     )

#     return {
#         **result,
#         "output_folder": output_folder if save_annotated else None,
#         "log": log_events,
#     }


# @router.get("/identities")
# def list_searchable_identities():
#     """
#     List all identities currently available for searching.
#     """
#     store = get_store()
#     identities = store.list_identities()
#     return {
#         "total": len(identities),
#         "identities": [i["name"] for i in identities],
#     }

"""
Search API
----------
Routes for searching by identity across image collections.

POST /search/person        → Search for one person across all test images
POST /search/index-all     → Index all persons across all test images
GET  /search/persons       → List all persons found in test images (from last index)
"""

import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from app.core.config import get_settings, get_store, get_searcher

router = APIRouter(prefix="/search", tags=["Search"])


class PersonSearchRequest(BaseModel):
    name: str
    image_folder: Optional[str] = None
    save_annotated: bool = True


@router.post("/person")
def search_person(request: PersonSearchRequest):
    """
    Search for a specific person across an image collection.

    Returns all images where this person's face is detected,
    regardless of whether the image contains other faces.
    """
    settings = get_settings()
    store = get_store()
    searcher = get_searcher()

    reference_embeddings = store.get_all_embeddings()
    if not reference_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No reference embeddings. Run /embeddings/rebuild first."
        )

    if request.name not in reference_embeddings:
        available = list(reference_embeddings.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Identity '{request.name}' not found. Available: {available}"
        )

    image_folder = request.image_folder or settings.test_images_path

    if not os.path.isdir(image_folder):
        raise HTTPException(
            status_code=404,
            detail=f"Image folder not found: {image_folder}"
        )

    output_folder = os.path.join(
        settings.output_path,
        "search",
        request.name.replace(" ", "_"),
    )

    log_events = []
    def capture(event, payload):
        log_events.append({"event": event, "data": payload})

    result = searcher.search_person(
        query_name=request.name,
        image_folder=image_folder,
        reference_embeddings=reference_embeddings,
        output_folder=output_folder,
        save_annotated=request.save_annotated,
        progress_callback=capture,
    )

    return {
        **result,
        "output_folder": output_folder if request.save_annotated else None,
        "log": log_events,
    }


@router.post("/index-all")
def index_all(
    image_folder: Optional[str] = Query(None),
    save_annotated: bool = Query(True),
):
    """
    Process ALL images and build a full index of who appears where.
    Returns a person → [image list] mapping.
    """
    settings = get_settings()
    store = get_store()
    searcher = get_searcher()

    reference_embeddings = store.get_all_embeddings()
    if not reference_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No reference embeddings. Run /embeddings/rebuild first."
        )

    target_folder = image_folder or settings.test_images_path
    if not os.path.isdir(target_folder):
        raise HTTPException(
            status_code=404,
            detail=f"Image folder not found: {target_folder}"
        )

    output_folder = os.path.join(settings.output_path, "index")

    log_events = []
    def capture(event, payload):
        log_events.append({"event": event, "data": payload})

    result = searcher.search_all_persons(
        image_folder=target_folder,
        reference_embeddings=reference_embeddings,
        output_folder=output_folder,
        save_annotated=save_annotated,
        progress_callback=capture,
    )

    return {
        **result,
        "output_folder": output_folder if save_annotated else None,
        "log": log_events,
    }


@router.get("/identities")
def list_searchable_identities():
    """
    List all identities currently available for searching.
    """
    store = get_store()
    identities = store.list_identities()
    return {
        "total": len(identities),
        "identities": [i["name"] for i in identities],
    }