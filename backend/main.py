"""
Smart Needle — FastAPI Application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import get_settings, init_ml_components
from app.api.embeddings import router as embeddings_router
from app.api.recognition import router as recognition_router
from app.api.search import router as search_router



def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Smart Needle",
        description="Deep-learning facial similarity engine powered by ArcFace/InsightFace",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow frontend dev server
    # app.add_middleware(
    #     CORSMiddleware,
    #     allow_origins=["http://localhost:3000", "http://localhost:3001"],
    #     allow_credentials=True,
    #     allow_methods=["*"],
    #     allow_headers=["*"],
    # )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Mount output folder as static files (for serving annotated images)
    os.makedirs(settings.output_path, exist_ok=True)
    app.mount("/output", StaticFiles(directory=settings.output_path), name="output")

    # Routers
    app.include_router(embeddings_router, prefix="/api")
    app.include_router(recognition_router, prefix="/api")
    app.include_router(search_router, prefix="/api")

    @app.on_event("startup")
    async def startup():
        print("⚡ Loading InsightFace model...")
        init_ml_components()
        print("✅ Smart Needle ready.")

    @app.get("/api/health")
    def health():
        from app.core.config import get_store
        store = get_store()
        return {
            "status": "ok",
            "total_identities": store.total(),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)