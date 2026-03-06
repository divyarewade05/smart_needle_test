"""
Recognition API
---------------
Routes for running face recognition on images.

POST /recognize/image     → Upload single image, get all recognized faces
POST /recognize/folder    → Process all images in test folder
"""

import os
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import get_settings, get_engine, get_store, get_recognizer

router = APIRouter(prefix="/recognize", tags=["Recognition"])


@router.post("/image")
async def recognize_image(file: UploadFile = File(...)):
    """
    Upload a single image (can be multi-face, noisy, etc.).
    Returns all detected faces with matched identities and scores.
    Saves annotated image to output folder.
    """
    settings = get_settings()
    engine = get_engine()
    store = get_store()
    recognizer = get_recognizer()

    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    reference_embeddings = store.get_all_embeddings()

    if not reference_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No reference embeddings found. Run /embeddings/rebuild first."
        )

    # 0. Save original file to persistent storage (so feedback system can crop it)
    os.makedirs(settings.test_images_path, exist_ok=True)
    orig_save_path = os.path.join(settings.test_images_path, file.filename)
    with open(orig_save_path, "wb") as f:
        f.write(contents)

    # Detect all faces
    detected_faces = engine.detect_all_faces(img)

    if not detected_faces:
        return {
            "filename": file.filename,
            "faces_detected": 0,
            "recognized": [],
            "annotated_path": None,
        }

    # Match faces
    embeddings = [f["embedding"] for f in detected_faces]
    match_results = recognizer.match_batch(embeddings, reference_embeddings, top_k=3)

    recognized = []
    annotations = []

    for face, match in zip(detected_faces, match_results):
        recognized.append({
            "bbox": face["bbox"],
            "det_score": face["det_score"],
            "name": match["name"],
            "score": match["score"],
            "matched": match["matched"],
            "top_k": match["top_k"],
        })
        annotations.append({
            "bbox": face["bbox"],
            "name": match["name"],
            "score": match["score"],
            "matched": match["matched"],
        })

    os.makedirs(settings.output_path, exist_ok=True)
    # Also ensure reference and test paths exist in the new storage
    os.makedirs(settings.reference_path, exist_ok=True)
    os.makedirs(settings.test_images_path, exist_ok=True)
    
    annotated = engine.annotate_image(img, annotations)
    output_filename = f"recognized_{file.filename}"
    output_path = os.path.join(settings.output_path, output_filename)
    cv2.imwrite(output_path, annotated)

    return {
        "filename": file.filename,
        "faces_detected": len(detected_faces),
        "recognized": recognized,
        "annotated_path": output_filename,
    }


@router.post("/folder")
def recognize_folder(folder_path: str = None):
    """
    Process all images in the configured test folder.
    Returns per-image recognition results.
    """
    settings = get_settings()
    engine = get_engine()
    store = get_store()
    recognizer = get_recognizer()

    target_folder = folder_path or settings.test_images_path

    if not os.path.isdir(target_folder):
        raise HTTPException(status_code=404, detail=f"Folder not found: {target_folder}")

    reference_embeddings = store.get_all_embeddings()
    if not reference_embeddings:
        raise HTTPException(
            status_code=422,
            detail="No reference embeddings found. Run /embeddings/rebuild first."
        )

    from app.ml.embedding_store import SUPPORTED_EXTS
    from pathlib import Path

    os.makedirs(settings.output_path, exist_ok=True)

    results = []
    image_files = [
        f for f in Path(target_folder).iterdir()
        if f.suffix.lower() in SUPPORTED_EXTS
    ]

    for img_path in sorted(image_files):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        detected_faces = engine.detect_all_faces(img)

        if not detected_faces:
            results.append({
                "image": img_path.name,
                "faces_detected": 0,
                "recognized": [],
                "annotated_path": None,
            })
            continue

        embeddings = [f["embedding"] for f in detected_faces]
        match_results = recognizer.match_batch(embeddings, reference_embeddings)

        recognized = []
        annotations = []
        for face, match in zip(detected_faces, match_results):
            recognized.append({
                "bbox": face["bbox"],
                "name": match["name"],
                "score": match["score"],
                "matched": match["matched"],
            })
            annotations.append({
                "bbox": face["bbox"],
                "name": match["name"],
                "score": match["score"],
                "matched": match["matched"],
            })

        annotated = engine.annotate_image(img, annotations)
        out_filename = f"recognized_{img_path.name}"
        out_path = os.path.join(settings.output_path, out_filename)
        cv2.imwrite(out_path, annotated)

        results.append({
            "image": img_path.name,
            "faces_detected": len(detected_faces),
            "recognized": recognized,
            "annotated_path": out_filename,
        })

    return {
        "folder": target_folder,
        "total_images": len(results),
        "results": results,
    }


