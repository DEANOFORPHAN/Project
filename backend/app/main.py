from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.rl_routes import router as rl_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "backend is running"}


app.include_router(rl_router)
