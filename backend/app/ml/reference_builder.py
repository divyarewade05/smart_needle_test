"""
Reference Embedding Builder
----------------------------
Builds and incrementally updates mean face embeddings from a
reference folder structured as:

    reference/
        person_name/
            photo1.jpg
            photo2.jpg
            ...

Only recomputes embeddings for persons whose folder has changed
(added, removed, or files modified). Clean persons are skipped.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from app.ml.embedding_store import EmbeddingStore, SUPPORTED_EXTS
from app.ml.face_engine import FaceEngine


def build_reference_embeddings(
    reference_path: str,
    store: EmbeddingStore,
    engine: FaceEngine,
    force_rebuild: bool = False,
    progress_callback: Optional[Callable[[str, dict], None]] = None,
) -> dict:
    """
    Incrementally rebuild reference embeddings.

    Args:
        reference_path:     Root folder containing one subdir per person.
        store:              EmbeddingStore instance (loaded from disk).
        engine:             FaceEngine for embedding extraction.
        force_rebuild:      Ignore hashes, recompute everyone.
        progress_callback:  Optional fn(event_name, payload) for streaming.

    Returns:
        Summary dict with counts of added/updated/removed/skipped.
    """

    def emit(event: str, payload: dict):
        if progress_callback:
            progress_callback(event, payload)

    # ------------------------------------------------------------------ #
    #  1. Detect what changed                                              #
    # ------------------------------------------------------------------ #
    if force_rebuild:
        dirty = {
            "added": [],
            "removed": [],
            "modified": [],
            "clean": [],
            "_current_hashes": {},
        }
        root = Path(reference_path)
        all_persons = [d.name for d in root.iterdir() if d.is_dir()]
        dirty["added"] = all_persons

        from app.ml.embedding_store import _hash_reference_root
        dirty["_current_hashes"] = _hash_reference_root(reference_path)
    else:
        dirty = store.get_dirty_persons(reference_path)

    to_process = dirty["added"] + dirty["modified"]
    to_remove = dirty["removed"]
    current_hashes = dirty["_current_hashes"]

    emit("scan_complete", {
        "added": len(dirty["added"]),
        "modified": len(dirty["modified"]),
        "removed": len(dirty["removed"]),
        "clean": len(dirty["clean"]),
    })

    summary = {
        "processed": 0,
        "removed": 0,
        "skipped": len(dirty["clean"]),
        "failed": 0,
        "errors": [],
    }

    # ------------------------------------------------------------------ #
    #  2. Remove deleted persons                                           #
    # ------------------------------------------------------------------ #
    for person_name in to_remove:
        store.remove(person_name)
        summary["removed"] += 1
        emit("identity_removed", {"name": person_name})

    # ------------------------------------------------------------------ #
    #  3. Process dirty persons                                            #
    # ------------------------------------------------------------------ #
    for person_name in to_process:
        person_folder = os.path.join(reference_path, person_name)
        emit("identity_start", {"name": person_name})

        person_embeddings = []
        errors = []

        for file in sorted(os.listdir(person_folder)):
            if Path(file).suffix.lower() not in SUPPORTED_EXTS:
                continue

            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path)

            if img is None:
                msg = f"{person_name}/{file}: could not read"
                errors.append(msg)
                continue

            # Extract embedding - reference images should be clean single-face
            result = engine.extract_single_face_embedding(img)

            if result is None:
                msg = f"{person_name}/{file}: no face detected"
                errors.append(msg)
                continue

            if result["multiple_faces"]:
                msg = f"{person_name}/{file}: multiple faces — using largest"
                errors.append(msg)  # log as warning, not hard error

            person_embeddings.append(result["embedding"])

        if not person_embeddings:
            msg = f"{person_name}: zero valid embeddings — skipping"
            summary["errors"].append(msg)
            summary["failed"] += 1
            emit("identity_failed", {"name": person_name, "reason": msg})
            continue

        # Compute mean embedding, re-normalize
        mean_emb = np.mean(person_embeddings, axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)

        folder_hash = current_hashes.get(person_name, "")
        store.upsert(
            person_name=person_name,
            mean_embedding=mean_emb,
            sample_count=len(person_embeddings),
            folder_hash=folder_hash,
        )

        summary["processed"] += 1
        emit("identity_done", {
            "name": person_name,
            "samples": len(person_embeddings),
            "warnings": errors,
        })

    # ------------------------------------------------------------------ #
    #  4. Commit new hashes and save                                       #
    # ------------------------------------------------------------------ #
    store.commit_hashes(current_hashes)
    store.save()

    emit("build_complete", {**summary, "total": store.total()})
    return summary