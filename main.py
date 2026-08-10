from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import requests
from dotenv import load_dotenv
from scipy.spatial.distance import cdist

from models.Pupil import Pupil
from models.prediction.PredictionResponse import PredictionResponse
from services.ParamsExtractorService import ParamsExtractorService

load_dotenv()

app = FastAPI(title="Career Prediction API")


# --- Load models on startup ---
MODEL_DIR = "data"
TIMESTAMP_FILE = os.path.join(MODEL_DIR, "timestamp.txt")
# --- Init services ---
extractor = ParamsExtractorService()
def load_latest_models():
    """Load the latest clustering models"""
    with open(TIMESTAMP_FILE, 'r') as f:
        timestamp = f.readlines()[-1].strip()
    
    run_dir = os.path.join(MODEL_DIR, timestamp)
    
    # Load scaler
    with open(os.path.join(run_dir, 'clustering_results_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    
    # Load PCA
    with open(os.path.join(run_dir, 'clustering_results_pca_2d.pkl'), 'rb') as f:
        pca = pickle.load(f)
    
    # Load KMeans (using best K - let's say K=5 as example)
    # In real implementation, pick best K from metrics
    best_k = 5
    with open(os.path.join(run_dir, f'clustering_results_kmeans_k{best_k}.pkl'), 'rb') as f:
        kmeans = pickle.load(f)
    
    # Load specialist data
    X_scaled = np.load(os.path.join(run_dir, 'clustering_results_X_scaled.npy'))
    professions = np.load(os.path.join(run_dir, 'clustering_results_professions.npy'), allow_pickle=True)
    spec_ids = pd.read_pickle(os.path.join(run_dir, 'clustering_results_df_spec.pkl'))['Id'].values
    
    # Load feature columns
    feature_cols = np.load(os.path.join(run_dir, 'clustering_results_feature_cols.npy'), allow_pickle=True)
    
    return {
        'scaler': scaler,
        'pca': pca,
        'kmeans': kmeans,
        'X_spec_scaled': X_scaled,
        'professions': professions,
        'spec_ids': spec_ids,
        'feature_cols': feature_cols,
        'best_k': best_k
    }

# Load models at startup
models = load_latest_models()

# --- Distance categories ---
def load_categories():
    categories = []
    with open('category.csv', 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                min_val = float(parts[0])
                max_val = float('inf') if parts[1] == 'inf' else float(parts[1])
                categories.append({
                    'min': min_val, 
                    'max': max_val, 
                    'category': parts[2]
                })
    return categories

distance_categories = load_categories()

def get_category(distance):
    for cat in distance_categories:
        if cat['min'] <= distance <= cat['max']:
            return cat['category']
    return "Не определено"

# --- Prediction endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_pupil(pupil: Pupil):
    """
    Predict profession for a single pupil
    """
    try:
        print(models['feature_cols'])
        # 1. Convert to feature vector
        feature_values = extractor.extract_features(pupil, models['feature_cols'])
        
        X_pupil = np.array(feature_values).reshape(1, -1)
        
        # 2. Scale
        X_pupil_scaled = models['scaler'].transform(X_pupil)
        
        # 3. Predict cluster
        cluster = models['kmeans'].predict(X_pupil_scaled)[0]
        
        # 4. Find nearest specialist in that cluster
        labels = models['kmeans'].labels_
        indices_in_cluster = np.where(labels == cluster)[0]
        
        distances = cdist(X_pupil_scaled, models['X_spec_scaled'][indices_in_cluster])[0]
        nearest_idx = np.argmin(distances)
        nearest_spec_position = indices_in_cluster[nearest_idx]
        
        nearest_distance = distances[nearest_idx]
        predicted_profession = models['professions'][nearest_spec_position]
        nearest_spec_id = models['spec_ids'][nearest_spec_position]
        
        # 5. Send to API (optional)
        #await send_to_api(pupil.id, cluster, predicted_profession)
        
        return PredictionResponse(
            pupilId=pupil.pupilId,
            cluster=int(cluster),
            predictedProfession=str(predicted_profession),
            nearestSpecialistId=int(nearest_spec_id),
            distance=float(nearest_distance),
            confidenceCategory=get_category(nearest_distance)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def send_to_api(pupil_id, cluster, profession):
    """Send prediction to external API"""
    url = f"{os.getenv('API_BASE_URL')}/api/predictions/create"
    token = get_token()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    data = {
        "pupilId": pupil_id,
        "predictionType": "CLUSTER",
        "cluster": int(cluster),
        "predictedProfession": profession
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        return response.status_code == 200
    except:
        return False

def get_token():
    """Get auth token"""
    url = f"{os.getenv('API_BASE_URL')}/api/auth/login"
    data = {
        "email": os.getenv("API_EMAIL"),
        "password": os.getenv("API_PASSWORD")
    }
    response = requests.post(url, json=data, timeout=10)
    return response.json().get('token')

# --- Health check ---
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "best_k": models['best_k'],
        "features": list(models['feature_cols'])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)