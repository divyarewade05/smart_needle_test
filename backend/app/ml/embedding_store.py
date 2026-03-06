"""
Smart Embedding Store
---------------------
Hash-based change detection: only recomputes embeddings when the
reference folder actually changes (files added, removed, or modified).
Persists to a structured pickle file with metadata.
"""

import os
import hashlib
import pickle
import time
import numpy as np
from pathlib import Path
from typing import Optional


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _hash_folder(folder_path: str) -> str:
    """
    Compute a deterministic hash of a person's folder.
    Hash is based on: filenames + file sizes + last-modified times.
    Fast — does NOT read image bytes.
    """
    hasher = hashlib.md5()
    folder = Path(folder_path)

    entries = sorted(folder.iterdir())  # sorted for determinism

    for entry in entries:
        if entry.suffix.lower() not in SUPPORTED_EXTS:
            continue
        stat = entry.stat()
        hasher.update(entry.name.encode())
        hasher.update(str(stat.st_size).encode())
        hasher.update(str(int(stat.st_mtime)).encode()) # int() for robust hash

    return hasher.hexdigest()


def _hash_reference_root(reference_path: str) -> dict:
    """
    Returns a dict of {person_name: folder_hash} for all subdirs.
    """
    hashes = {}
    root = Path(reference_path)

    for person_dir in sorted(root.iterdir()):
        if person_dir.is_dir():
            hashes[person_dir.name] = _hash_folder(str(person_dir))

    return hashes


class EmbeddingStore:
    """
    Manages face embeddings with smart dirty-tracking.

    Store structure (pickle):
    {
        "embeddings": {
            "person_name": {
                "mean_embedding": np.ndarray (512,),
                "sample_count": int,
                "folder_hash": str,
                "updated_at": float (unix timestamp)
            }
        },
        "root_hash": dict  # {person_name: folder_hash} snapshot
    }
    """

    def __init__(self, store_path: str = "embeddings.pkl"):
        self.store_path = store_path
        self._data = self._load()

    # ------------------------------------------------------------------ #
    #  Load / Save                                                         #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict:
        if os.path.exists(self.store_path):
            with open(self.store_path, "rb") as f:
                return pickle.load(f)
        return {"embeddings": {}, "root_hash": {}}

    def save(self):
        with open(self.store_path, "wb") as f:
            pickle.dump(self._data, f)

    # ------------------------------------------------------------------ #
    #  Change Detection                                                    #
    # ------------------------------------------------------------------ #

    def get_dirty_persons(self, reference_path: str) -> dict:
        """
        Compare current folder state against stored hashes.

        Returns:
            {
                "added":   [person_names],   # new folders not in store
                "removed": [person_names],   # folders deleted from disk
                "modified":[person_names],   # folders whose hash changed
                "clean":   [person_names],   # no changes
            }
        """
        current_hashes = _hash_reference_root(reference_path)
        stored_hashes = self._data.get("root_hash", {})

        current_names = set(current_hashes.keys())
        stored_names = set(stored_hashes.keys())

        added = list(current_names - stored_names)
        removed = list(stored_names - current_names)
        modified = [
            name for name in current_names & stored_names
            if current_hashes[name] != stored_hashes[name]
        ]
        clean = [
            name for name in current_names & stored_names
            if current_hashes[name] == stored_hashes[name]
        ]

        return {
            "added": added,
            "removed": removed,
            "modified": modified,
            "clean": clean,
            "_current_hashes": current_hashes,  # carry forward for update
        }

    # ------------------------------------------------------------------ #
    #  Write Operations                                                    #
    # ------------------------------------------------------------------ #

    def upsert(
        self,
        person_name: str,
        mean_embedding: np.ndarray,
        sample_count: int,
        folder_hash: str,
    ):
        """Insert or update a person's embedding entry."""
        self._data["embeddings"][person_name] = {
            "mean_embedding": mean_embedding,
            "sample_count": sample_count,
            "folder_hash": folder_hash,
            "updated_at": time.time(),
        }

    def remove(self, person_name: str):
        """Remove a person from the store."""
        self._data["embeddings"].pop(person_name, None)
        self._data["root_hash"].pop(person_name, None)

    def commit_hashes(self, new_hashes: dict):
        """Persist the latest folder hashes after a successful rebuild."""
        self._data["root_hash"] = new_hashes

    # ------------------------------------------------------------------ #
    #  Read Operations                                                     #
    # ------------------------------------------------------------------ #

    def get_all_embeddings(self) -> dict:
        """Returns {person_name: np.ndarray (512,)} for similarity search."""
        return {
            name: entry["mean_embedding"]
            for name, entry in self._data["embeddings"].items()
        }

    def get_person(self, person_name: str) -> Optional[dict]:
        return self._data["embeddings"].get(person_name)

    def list_identities(self) -> list:
        """Return metadata for all stored identities (no raw embeddings)."""
        result = []
        for name, entry in self._data["embeddings"].items():
            result.append({
                "name": name,
                "sample_count": entry["sample_count"],
                "updated_at": entry["updated_at"],
            })
        return sorted(result, key=lambda x: x["name"])

    def total(self) -> int:
        return len(self._data["embeddings"])