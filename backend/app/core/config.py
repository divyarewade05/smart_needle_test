# """
# Core Config & Dependencies
# --------------------------
# Single source of truth for paths, thresholds, and shared ML singletons.
# """

# import os
# from functools import lru_cache
# from pydantic_settings import BaseSettings
# from app.ml.face_engine import FaceEngine
# from app.ml.embedding_store import EmbeddingStore
# from app.ml.recognition_service import RecognitionService
# from app.ml.search_service import SearchService


# class Settings(BaseSettings):
#     # Paths
#     reference_path: str = os.getenv("REFERENCE_PATH", "data/reference")
#     test_images_path: str = os.getenv("TEST_IMAGES_PATH", "data/test")
#     output_path: str = os.getenv("OUTPUT_PATH", "data/output")
#     embeddings_path: str  = os.getenv("EMBEDDINGS_PATH", "embeddings.pkl")

#     # Model
#     insightface_model: str = "buffalo_l"
#     ctx_id: int = -1                    # -1 = CPU
#     det_size: tuple = (640, 640)
#     det_score_thresh: float = 0.5

#     # Recognition
#     similarity_threshold: float = 0.45  # cosine similarity cutoff

#     class Config:
#         env_file = ".env"


# @lru_cache()
# def get_settings() -> Settings:
#     return Settings()


# # ------------------------------------------------------------------ #
# #  ML Singletons — loaded once at startup                             #
# # ------------------------------------------------------------------ #

# _engine: FaceEngine = None
# _store: EmbeddingStore = None
# _recognizer: RecognitionService = None
# _searcher: SearchService = None


# def init_ml_components():
#     """Call this once on FastAPI startup."""
#     global _engine, _store, _recognizer, _searcher

#     settings = get_settings()

#     _engine = FaceEngine(
#         model_name=settings.insightface_model,
#         ctx_id=settings.ctx_id,
#         det_size=settings.det_size,
#         det_score_thresh=settings.det_score_thresh,
#     )
#     _engine.load()

#     # _store = EmbeddingStore(store_path=settings.embeddings_store_path)
#     _store = EmbeddingStore(store_path=settings.embeddings_path)

#     _recognizer = RecognitionService(threshold=settings.similarity_threshold)

#     _searcher = SearchService(engine=_engine, recognizer=_recognizer)


# def get_engine() -> FaceEngine:
#     return _engine


# def get_store() -> EmbeddingStore:
#     return _store


# def get_recognizer() -> RecognitionService:
#     return _recognizer


# def get_searcher() -> SearchService:
#     return _searcher

"""
Core Config & Dependencies
"""

import os
from app.ml.face_engine import FaceEngine
from app.ml.embedding_store import EmbeddingStore
from app.ml.recognition_service import RecognitionService
from app.ml.search_service import SearchService

# ── Absolute paths ──────────────────────────────────────────────────
# We move 'data' outside of the 'backend' folder to prevent Uvicorn --reload from restarting 
# every time a result image is saved.
_ROOT = r"C:\Users\Administrator\Desktop\Smart-Needle"
_STORAGE = os.path.join(_ROOT, "data_storage")

REFERENCE_PATH      = os.path.join(_STORAGE, "reference")
TEST_IMAGES_PATH    = os.path.join(_STORAGE, "test")
OUTPUT_PATH         = os.path.join(_STORAGE, "output")
EMBEDDINGS_PATH     = os.path.join(_STORAGE, "embeddings.pkl")

INSIGHTFACE_MODEL   = "buffalo_l"
CTX_ID              = -1
DET_SIZE            = (640, 640)
DET_SCORE_THRESH    = 0.5
SIMILARITY_THRESHOLD = 0.30


class Settings:
    reference_path       = REFERENCE_PATH
    test_images_path     = TEST_IMAGES_PATH
    output_path          = OUTPUT_PATH
    embeddings_store_path = EMBEDDINGS_PATH
    insightface_model    = INSIGHTFACE_MODEL
    ctx_id               = CTX_ID
    det_size             = DET_SIZE
    det_score_thresh     = DET_SCORE_THRESH
    similarity_threshold = SIMILARITY_THRESHOLD


def get_settings():
    s = Settings()
    print(f"Reference:  {s.reference_path}")
    print(f"Test images: {s.test_images_path}")
    print(f"Output:     {s.output_path}")
    return s


# ── ML Singletons ───────────────────────────────────────────────────
_engine     = None
_store      = None
_recognizer = None
_searcher   = None


def init_ml_components():
    global _engine, _store, _recognizer, _searcher
    s = get_settings()

    _engine = FaceEngine(
        model_name=s.insightface_model,
        ctx_id=s.ctx_id,
        det_size=s.det_size,
        det_score_thresh=s.det_score_thresh,
    )
    _engine.load()

    _store      = EmbeddingStore(store_path=s.embeddings_store_path)
    _recognizer = RecognitionService(threshold=s.similarity_threshold)
    _searcher   = SearchService(engine=_engine, recognizer=_recognizer)


def get_engine():     return _engine
def get_store():      return _store
def get_recognizer(): return _recognizer
def get_searcher():   return _searcher