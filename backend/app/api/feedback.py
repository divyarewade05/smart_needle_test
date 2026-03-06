"""
Feedback & Active Learning API
------------------------------
Handles teacher/parent verification feedback.
Converts verified recognition results into new reference images.
"""

import os
import cv2
import numpy as np
import time
from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
from typing import List, Dict

from app.core.config import get_settings, get_store

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.get("/reference/{person_name}")
async def get_reference_sample(person_name: str):
    """Returns the most appropriate reference image for a person."""
    settings = get_settings()
    person_path = os.path.join(settings.reference_path, person_name)
    
    if not os.path.isdir(person_path):
        raise HTTPException(status_code=404, detail=f"Person '{person_name}' not found.")
    
    # Normalize target for robust search
    target = person_name.strip().lower()
    
    # Filter only supported image files
    from app.ml.embedding_store import SUPPORTED_EXTS
    valid_files = [f for f in os.listdir(person_path) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS]
    
    if not valid_files:
        raise HTTPException(status_code=404, detail="Identity exists but has no imagery.")

    # Sort files by size (descending) as a heuristic - original photos are almost always larger than crops
    # and also filter out known feedback patterns
    candidates = []
    for f in valid_files:
        if f.lower().startswith("feedback_"):
            continue
        full_p = os.path.join(person_path, f)
        candidates.append((f, os.path.getsize(full_p)))
        
    # Priority 1: Exact name match (ignoring extension)
    master_file = None
    for f, size in candidates:
        if os.path.splitext(f)[0].lower() == target:
            master_file = f
            break
            
    if master_file:
        return FileResponse(os.path.join(person_path, master_file))

    # Priority 2: Return the largest remaining non-feedback file
    if candidates:
        # Sort by size (idx 1 of tuple) descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return FileResponse(os.path.join(person_path, candidates[0][0]))
    
    # Priority 3: Absolute fallback - first valid file
    return FileResponse(os.path.join(person_path, valid_files[0]))

@router.get("/crop")
async def get_face_crop(filename: str, x1: int, y1: int, x2: int, y2: int):
    """Returns a temporary crop of a face for UI preview."""
    settings = get_settings()
    
    src_path = os.path.join(settings.output_path, filename)
    if not os.path.exists(src_path):
        alt_name = filename.replace("recognized_", "", 1) if filename.startswith("recognized_") else filename
        src_path = os.path.join(settings.test_images_path, alt_name)

    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail="Source image not found.")

    img = cv2.imread(src_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Failed to read image.")

    # Crop
    h, w = img.shape[:2]
    # Add 15% padding
    pw, ph = int((x2-x1)*0.15), int((y2-y1)*0.15)
    cx1, cy1 = max(0, x1-pw), max(0, y1-ph)
    cx2, cy2 = min(w, x2+pw), min(h, y2+ph)
    
    crop = img[cy1:cy2, cx1:cx2]
    
    # Encode to memory
    _, buffer = cv2.imencode(".jpg", crop)
    from fastapi.responses import Response
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@router.post("/submit")
async def submit_feedback(
    image_filename: str = Body(...),
    verifications: List[Dict] = Body(...)
):
    """
    Receives feedback from the teacher.
    Each verification dict: {
        "bbox": [x1, y1, x2, y2],
        "is_correct": bool,
        "corrected_name": str (optional)
    }
    """
    settings = get_settings()
    store = get_store()
    
    # Locate the original uploaded image in output or test folder
    # For this test, we assume the file is in the output path (where recognized images are saved temporarily)
    # The frontend should send the original filename
    
    # Note: In a production app, we'd store the original upload in a tmp folder.
    # For now, let's look in the test folder or assume the frontend provides a way to get the source.
    # To keep it simple for the 'Teacher Test', we will ask the frontend to send the image base64 or path.
    
    # Actually, let's simplify: the frontend sends the crop itself if it's verified.
    # OR the frontend sends the original filename and we find it.
    
    # For the "Teacher Test", let's implement the 'Corrected Name' -> 'Save Crop' logic.
    
    processed_count = 0
    
    # We need the source image to crop the face
    # Ensure source image is found in data_storage/output or test
    src_path = os.path.join(settings.output_path, image_filename)
    if not os.path.exists(src_path):
        # Maybe it's without 'recognized_' prefix
        alt_name = image_filename.replace("recognized_", "", 1) if image_filename.startswith("recognized_") else image_filename
        src_path = os.path.join(settings.test_images_path, alt_name)

    if not os.path.exists(src_path):
         return {"status": "error", "message": f"Source image {image_filename} not found in {settings.output_path} or {settings.test_images_path}"}

    img = cv2.imread(src_path)
    if img is None:
        return {"status": "error", "message": "Failed to read source image for cropping."}

    for v in verifications:
        if v.get("is_correct") or v.get("corrected_name"):
            target_name = v.get("corrected_name") or v.get("name")
            if not target_name or target_name == "Unknown":
                continue
            
            # Crop the face
            x1, y1, x2, y2 = v["bbox"]
            # Add small padding
            h, w = img.shape[:2]
            pw, ph = int((x2-x1)*0.1), int((y2-y1)*0.1)
            x1_p, y1_p = max(0, x1-pw), max(0, y1-ph)
            x2_p, y2_p = min(w, x2+pw), min(h, y2+ph)
            
            face_crop = img[y1_p:y2_p, x1_p:x2_p]
            
            # Save to reference folder
            target_dir = os.path.join(settings.reference_path, target_name)
            os.makedirs(target_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"feedback_{timestamp}_{processed_count}.jpg"
            save_path = os.path.join(target_dir, filename)
            
            cv2.imwrite(save_path, face_crop)
            processed_count += 1

    if processed_count > 0:
        # 4. Instant Intelligence Update: Recalculate mean embeddings for the modified students
        # The builder is smart: it will only re-process folders whose hash has changed (the ones we just added photos to)
        from app.ml.reference_builder import build_reference_embeddings
        from app.core.config import get_engine
        
        # build_reference_embeddings is now a generator
        for _ in build_reference_embeddings(
            reference_path=settings.reference_path,
            store=store,
            engine=get_engine(),
            force_rebuild=False
        ):
            pass

    return {
        "status": "ok",
        "message": f"Successfully processed {processed_count} feedback items. AI Core updated.",
        "reward_applied": True,
        "active_learning_updated": processed_count > 0
    }
