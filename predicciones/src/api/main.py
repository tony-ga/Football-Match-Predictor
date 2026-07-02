from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import predict

app = FastAPI(
    title="Football Match Prediction API",
    description="API for comprehensive football match predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
