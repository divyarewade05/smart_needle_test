"""
Search Service
--------------
Given a query (person name or ID), scans an image collection and
returns every image where that person's face is detected and recognized.

Supports:
  - Single-face and multi-face images
  - Noisy / cluttered scenes
  - Returns matched image paths + face locations + similarity scores
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from app.ml.face_engine import FaceEngine
from app.ml.recognition_service import RecognitionService
from app.ml.embedding_store import SUPPORTED_EXTS


class SearchService:
    """
    Scans a folder of images and finds all appearances of a target identity.
    """

    def __init__(self, engine: FaceEngine, recognizer: RecognitionService):
        self.engine = engine
        self.recognizer = recognizer

    def search_person(
        self,
        query_name: str,
        image_folder: str,
        reference_embeddings: dict,        # {name: np.ndarray}
        output_folder: Optional[str] = None,
        save_annotated: bool = True,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """
        Search for a person by name across all images in image_folder.

        Args:
            query_name:            The identity name to search for.
            image_folder:          Folder of scene images to scan.
            reference_embeddings:  All stored mean embeddings.
            output_folder:         Where to save annotated result images.
            save_annotated:        Whether to write annotated images to disk.
            progress_callback:     Optional fn(event, payload).

        Returns:
            {
                "query": str,
                "total_images_scanned": int,
                "matched_images": [
                    {
                        "image_path": str,
                        "annotated_path": str | None,
                        "faces_in_image": int,
                        "matched_faces": [
                            {
                                "bbox": [x1,y1,x2,y2],
                                "score": float,
                                "name": str,
                            }
                        ]
                    }
                ],
                "total_matches": int,   # total face-level matches
            }
        """

        def emit(event, payload):
            if progress_callback:
                progress_callback(event, payload)

        # Validate query name exists in references
        if query_name not in reference_embeddings:
            return {
                "query": query_name,
                "error": f"'{query_name}' not found in reference embeddings.",
                "matched_images": [],
                "total_images_scanned": 0,
                "total_matches": 0,
            }

        if output_folder and save_annotated:
            os.makedirs(output_folder, exist_ok=True)

        image_files = [
            f for f in Path(image_folder).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ]

        emit("search_start", {"query": query_name, "total_files": len(image_files)})

        matched_images = []
        total_face_matches = 0

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                emit("image_skipped", {"path": str(img_path), "reason": "unreadable"})
                continue

            # Detect all faces in the scene image
            detected_faces = self.engine.detect_all_faces(img)

            if not detected_faces:
                emit("image_processed", {"path": img_path.name, "faces": 0, "matched": False})
                continue

            # Match every detected face
            embeddings = [f["embedding"] for f in detected_faces]
            match_results = self.recognizer.match_batch(
                embeddings, reference_embeddings, top_k=3
            )

            # Filter: only faces matching the query person
            query_matches = []
            all_face_annotations = []

            for face, match in zip(detected_faces, match_results):
                is_query_match = (match["matched"] and match["name"] == query_name)

                all_face_annotations.append({
                    "bbox": face["bbox"],
                    "name": match["name"],
                    "score": match["score"],
                    "matched": match["matched"],
                })

                if is_query_match:
                    query_matches.append({
                        "bbox": face["bbox"],
                        "score": match["score"],
                        "name": match["name"],
                    })

            if not query_matches:
                emit("image_processed", {"path": img_path.name, "faces": len(detected_faces), "matched": False})
                continue

            # This image contains the query person
            total_face_matches += len(query_matches)

            annotated_path = None
            if save_annotated and output_folder:
                annotated_img = self.engine.annotate_image(img, all_face_annotations)
                annotated_path = os.path.join(output_folder, f"match_{img_path.name}")
                cv2.imwrite(annotated_path, annotated_img)

            matched_images.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "annotated_path": annotated_path,
                "faces_in_image": len(detected_faces),
                "matched_faces": query_matches,
            })

            emit("image_matched", {
                "path": img_path.name,
                "faces": len(detected_faces),
                "query_matches": len(query_matches),
            })

        emit("search_complete", {
            "query": query_name,
            "matched_images": len(matched_images),
            "total_matches": total_face_matches,
        })

        return {
            "query": query_name,
            "total_images_scanned": len(image_files),
            "matched_images": matched_images,
            "total_matches": total_face_matches,
        }

    def search_all_persons(
        self,
        image_folder: str,
        reference_embeddings: dict,
        output_folder: Optional[str] = None,
        save_annotated: bool = True,
        progress_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> dict:
        """
        Process all images and annotate every recognized face.
        Returns a full index: which person appears in which images.

        Returns:
            {
                "person_index": {
                    "person_name": ["image1.jpg", "image2.jpg", ...]
                },
                "image_results": [
                    {image_path, annotated_path, faces_in_image, recognized_faces}
                ]
            }
        """

        def emit(event, payload):
            if progress_callback:
                progress_callback(event, payload)

        if output_folder and save_annotated:
            os.makedirs(output_folder, exist_ok=True)

        image_files = [
            f for f in Path(image_folder).iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS
        ]

        emit("index_start", {"total_files": len(image_files)})

        person_index: dict[str, list[str]] = {}
        image_results = []

        for img_path in sorted(image_files):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            detected_faces = self.engine.detect_all_faces(img)

            if not detected_faces:
                image_results.append({
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "annotated_path": None,
                    "faces_in_image": 0,
                    "recognized_faces": [],
                })
                continue

            embeddings = [f["embedding"] for f in detected_faces]
            match_results = self.recognizer.match_batch(embeddings, reference_embeddings)

            annotations = []
            for face, match in zip(detected_faces, match_results):
                annotations.append({
                    "bbox": face["bbox"],
                    "name": match["name"],
                    "score": match["score"],
                    "matched": match["matched"],
                })

                # Update person index
                if match["matched"]:
                    person_name = match["name"]
                    if person_name not in person_index:
                        person_index[person_name] = []
                    if img_path.name not in person_index[person_name]:
                        person_index[person_name].append(img_path.name)

            annotated_path = None
            if save_annotated and output_folder:
                annotated_img = self.engine.annotate_image(img, annotations)
                annotated_path = os.path.join(output_folder, f"indexed_{img_path.name}")
                cv2.imwrite(annotated_path, annotated_img)

            image_results.append({
                "image_path": str(img_path),
                "image_name": img_path.name,
                "annotated_path": annotated_path,
                "faces_in_image": len(detected_faces),
                "recognized_faces": annotations,
            })

            emit("image_indexed", {"path": img_path.name, "faces": len(detected_faces)})

        emit("index_complete", {
            "total_persons_found": len(person_index),
            "total_images": len(image_results),
        })

        return {
            "person_index": person_index,
            "image_results": image_results,
        }