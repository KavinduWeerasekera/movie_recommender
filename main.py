from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = None
embeddings = None
index = None
model = None

class MovieRequest(BaseModel):
    movie_title: str
    k: int = 5

class MovieResponse(BaseModel):
    title: str
    listed_in: str
    description: str

app = FastAPI(title="Netflix Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_textual_representation(row):
    """Exactly from your notebook - no changes"""
    textual_representation = f"""Type:{row['type']},
Title:{row['title']},
Director:{row['director']},
Cast:{row['cast']},
Released:{row['release_year']},
Genres:{row['listed_in']},
Description:{row['description']}"""
    return textual_representation

def recommend_movies(movie_title, k=5):
    """Exactly from your notebook - no changes"""
    # Find index of the movie
    idx = df.index[df['title'] == movie_title].tolist()
    if not idx:
        return f"Movie '{movie_title}' not found."
    
    idx = idx[0]
    query_vector = embeddings[idx].reshape(1, -1)

    # Search in FAISS
    distances, indices = index.search(query_vector, k+1)  # k+1 to skip the movie itself
    
    # Get recommendations (skip the first one because it's the same movie)
    recommendations = df.iloc[indices[0][1:]][['title', 'listed_in', 'description']]
    return recommendations

@app.on_event("startup")
async def startup_event():
    global df, embeddings, index, model
    
    try:
        # Step 1: Load data 
        logger.info("Loading Netflix dataset...")
        df = pd.read_csv("netflix_titles.csv")
        
        # Step 2: Create textual representations 
        logger.info("Creating textual representations...")
        df['Textual_Representation'] = df.apply(create_textual_representation, axis=1)
        
        # Step 3: Load embedding model 
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Step 4: Encode all movies 
        logger.info("Creating embeddings...")
        embeddings = model.encode(df['Textual_Representation'].tolist(), convert_to_numpy=True)
        
        # Step 5: Normalize for cosine similarity 
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Step 6: Create FAISS index 
        index = faiss.IndexFlatIP(embeddings.shape[1])  
        index.add(embeddings)  
        
        logger.info(f"✅ Successfully loaded {len(df)} movies!")
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {str(e)}")

@app.get("/")
async def root():
    """Simple health check"""
    return {
        "message": "Netflix Recommender API", 
        "status": "ready" if df is not None else "loading",
        "total_movies": len(df) if df is not None else 0
    }

@app.get("/movies")
async def get_all_movies():
    """Get all movies for search dropdown"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    movies = []
    for _, row in df.iterrows():
        movies.append({
            "title": row['title'],
            "type": row['type'],
            "release_year": str(row['release_year']) if pd.notna(row['release_year']) else None,
            "listed_in": row['listed_in'] if pd.notna(row['listed_in']) else None,
            "description": row['description'] if pd.notna(row['description']) else None,
            "cast": row['cast'] if pd.notna(row['cast']) else None,
            "director": row['director'] if pd.notna(row['director']) else None,
        })
    
    return {"movies": movies}

@app.post("/recommend")
async def get_recommendations(request: MovieRequest):
    """Use your exact recommend_movies function"""
    if df is None or embeddings is None or index is None:
        raise HTTPException(status_code=503, detail="System not ready")
    result = recommend_movies(request.movie_title, request.k)
    
    if isinstance(result, str):
        raise HTTPException(status_code=404, detail=result)
    
    recommendations = []
    for _, row in result.iterrows():
        recommendations.append({
            "title": row['title'],
            "listed_in": row['listed_in'],
            "description": row['description']
        })
    
    return {"recommendations": recommendations}

@app.get("/search")
async def search_movies(q: str):
    """Simple search for frontend"""
    if df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet")
    
    # Simple search in title
    mask = df['title'].str.contains(q, case=False, na=False)
    results = df[mask].head(10)
    
    movies = []
    for _, row in results.iterrows():
        movies.append({
            "title": row['title'],
            "type": row['type'],
            "release_year": str(row['release_year']) if pd.notna(row['release_year']) else None,
            "listed_in": row['listed_in'] if pd.notna(row['listed_in']) else None,
            "description": row['description'] if pd.notna(row['description']) else None
        })
    
    return {"movies": movies}

if __name__ == "__main__":
    import uvicorn
    print("🎬 Starting Netflix Recommender API...")
    print("📁 Make sure netflix_titles.csv is in the current directory!")
    uvicorn.run(app, host="0.0.0.0", port=8000)