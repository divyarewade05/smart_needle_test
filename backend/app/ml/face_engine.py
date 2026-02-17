"""
Face Engine
-----------
Thin wrapper around InsightFace buffalo_l.

Responsibilities:
  - Model init (singleton, loaded once)
  - Single-face embedding extraction (for reference building)
  - Multi-face detection + recognition (for scene images)
  - Non-face / low-confidence filtering
"""

import cv2
import numpy as np
from typing import Optional
from insightface.app import FaceAnalysis


class FaceEngine:
    """
    Singleton-safe InsightFace wrapper.

    Usage:
        engine = FaceEngine()
        engine.load()  # call once at startup
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        ctx_id: int = -1,          # -1 = CPU, 0+ = GPU index
        det_size: tuple = (640, 640),
        det_score_thresh: float = 0.5,
    ):
        self.model_name = model_name
        self.ctx_id = ctx_id
        self.det_size = det_size
        self.det_score_thresh = det_score_thresh
        self._app: Optional[FaceAnalysis] = None

    def load(self):
        """Initialize InsightFace. Call once at app startup."""
        if self._app is not None:
            return  # already loaded

        self._app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(
            ctx_id=self.ctx_id,
            det_size=self.det_size,
            det_thresh=self.det_score_thresh,
        )

    # ------------------------------------------------------------------ #
    #  Reference image: expect single clean face                          #
    # ------------------------------------------------------------------ #

    def extract_single_face_embedding(self, img: np.ndarray) -> Optional[dict]:
        """
        For reference/ID images. Expects one dominant face.

        Returns:
            None if no face detected, else:
            {
                "embedding": np.ndarray (512,) — L2 normalized,
                "bbox": [x1, y1, x2, y2],
                "det_score": float,
                "multiple_faces": bool,
            }
        """
        faces = self._app.get(img)

        if not faces:
            return None

        # Pick highest-confidence face if multiple
        faces = sorted(faces, key=lambda f: f.det_score, reverse=True)
        face = faces[0]

        emb = face.embedding.astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        return {
            "embedding": emb,
            "bbox": face.bbox.astype(int).tolist(),
            "det_score": float(face.det_score),
            "multiple_faces": len(faces) > 1,
        }

    # ------------------------------------------------------------------ #
    #  Scene image: noisy, multi-face, may have non-face objects          #
    # ------------------------------------------------------------------ #

    def detect_all_faces(self, img: np.ndarray) -> list[dict]:
        """
        For scene/test images. Returns all detected faces.

        Each face dict:
        {
            "embedding": np.ndarray (512,) — L2 normalized,
            "bbox": [x1, y1, x2, y2],
            "det_score": float,
            "kps": np.ndarray — 5 keypoints (optional, for alignment viz),
        }

        Faces are sorted left → right (by x1) for consistent indexing.
        """
        faces = self._app.get(img)

        if not faces:
            return []

        result = []
        for face in faces:
            emb = face.embedding.astype(np.float32)
            emb = emb / np.linalg.norm(emb)

            result.append({
                "embedding": emb,
                "bbox": face.bbox.astype(int).tolist(),
                "det_score": float(face.det_score),
                "kps": face.kps.astype(int).tolist() if face.kps is not None else [],
            })

        # Sort left → right
        result.sort(key=lambda f: f["bbox"][0])
        return result

    # ------------------------------------------------------------------ #
    #  Annotate image with bounding boxes + labels                        #
    # ------------------------------------------------------------------ #

    def annotate_image(
        self,
        img: np.ndarray,
        recognized_faces: list[dict],
    ) -> np.ndarray:
        """
        Draw bounding boxes and name labels on image.

        recognized_faces: list of {
            "bbox": [x1,y1,x2,y2],
            "name": str,
            "score": float,
            "matched": bool,
        }

        Returns annotated image copy.
        """
        out = img.copy()

        for face in recognized_faces:
            x1, y1, x2, y2 = face["bbox"]
            name = face["name"]
            score = face["score"]
            matched = face["matched"]

            # Green = known identity, Red = unknown
            color = (0, 200, 0) if matched else (0, 0, 220)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label background
            label = f"{name} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

            # Label text
            cv2.putText(
                out,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out