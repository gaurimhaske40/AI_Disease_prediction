from fastapi import FastAPI
from .routers import diabetes, heart, kidney   # <-- notice the dot (relative import)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Disease Prediction API")

# ✅ Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:3000"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(diabetes.router, prefix="/diabetes", tags=["Diabetes"])
app.include_router(heart.router, prefix="/heart", tags=["Heart Disease"])
app.include_router(kidney.router, prefix="/kidney", tags=["Kidney Disease"])


@app.get("/")
def home():
    return {"message": "Welcome to AI Disease Prediction API"}
