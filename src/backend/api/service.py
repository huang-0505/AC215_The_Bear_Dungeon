from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routers import combat

# Setup FastAPI app
app = FastAPI(title="DnD Combat API", description="Backend API for DnD Combat Simulator", version="v1")

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/")
async def get_index():
    return {"message": "Welcome to DnD Combat API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Combat router
app.include_router(combat.router, prefix="/combat")
