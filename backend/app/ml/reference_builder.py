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
from typing import Optional

from app.ml.embedding_store import EmbeddingStore, SUPPORTED_EXTS
from app.ml.face_engine import FaceEngine


def build_reference_embeddings(
    reference_path: str,
    store: EmbeddingStore,
    engine: FaceEngine,
    force_rebuild: bool = False,
):
    """
    Incrementally rebuild reference embeddings as a generator.
    Yields events: {"event": str, "data": dict}
    """

    # 1. Detect what changed
    if force_rebuild:
        root = Path(reference_path)
        to_process = [d.name for d in root.iterdir() if d.is_dir()]
        to_remove = []
        clean_count = 0
        from app.ml.embedding_store import _hash_reference_root
        current_hashes = _hash_reference_root(reference_path)
    else:
        dirty = store.get_dirty_persons(reference_path)
        to_process = dirty["added"] + dirty["modified"]
        to_remove = dirty["removed"]
        clean_count = len(dirty["clean"])
        current_hashes = dirty["_current_hashes"]

    yield {"event": "scan_complete", "data": {
        "to_process": len(to_process),
        "removed": len(to_remove),
        "clean": clean_count,
    }}

    # Terminal feedback
    if len(to_process) > 0 or len(to_remove) > 0:
        print(f"🧠 SYNC START: Detected {len(to_process)} dirty folders.")
    else:
        print(f"🧠 SYNC: No changes found among {clean_count} identities.")

    summary = {"processed": 0, "removed": 0, "skipped": clean_count, "failed": 0, "errors": []}

    # 2. Remove deleted persons
    for person_name in to_remove:
        store.remove(person_name)
        summary["removed"] += 1
        yield {"event": "identity_removed", "data": {"name": person_name}}

    # 3. Process dirty ones
    batch_size = 10
    for i, person_name in enumerate(to_process):
        person_folder = os.path.join(reference_path, person_name)
        person_embeddings = []
        
        # Internal logging every few identities
        if i % batch_size == 0:
            print(f"🧠 Retraining Core... [{i}/{len(to_process)}]")

        for file in sorted(os.listdir(person_folder)):
            if Path(file).suffix.lower() not in SUPPORTED_EXTS: continue
            img_path = os.path.join(person_folder, file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            result = engine.extract_single_face_embedding(img)
            if result: person_embeddings.append(result["embedding"])

        if person_embeddings:
            mean_emb = np.mean(person_embeddings, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            store.upsert(person_name, mean_emb, len(person_embeddings), current_hashes.get(person_name, ""))
            summary["processed"] += 1
            yield {"event": "identity_done", "data": {"name": person_name, "count": len(person_embeddings)}}
        else:
            summary["failed"] += 1
            yield {"event": "identity_failed", "data": {"name": person_name}}

    # 4. Save and finish
    store.commit_hashes(current_hashes)
    store.save()
    
    final_summary = {**summary, "total": store.total()}
    print(f"🧠 SYNC COMPLETE: Processed {final_summary['processed']} identities.")
    yield {"event": "build_complete", "data": final_summary}