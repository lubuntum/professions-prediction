"""
Часть 2: Определение профессии пользователей методом ближайших соседей
======================================================================
Назначение:
    1. Загрузить результаты кластеризации из первой части
    2. Загрузить данные пользователей
    3. Для каждого пользователя и каждого K (2..N) определить:
       - Кластер, к которому относится пользователь
       - Ближайшего специалиста в этом кластере
       - Профессию ближайшего специалиста (прогноз)
    4. Сохранить результаты в CSV и JSON: одна строка на пользователя,
       с последовательными парами (K, cluster, idx, distance, profession)
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')
import get_parsing

# ======================== НАСТРОЙКИ ========================
# Дата и время
DATA_DIR = 'data'  # Папка для данных
OUTPUT_PREFIX = 'clustering_results'
OUTPUT_DIR_USER = "data_user"  # Папка для результатов учеников

# Загрузка категорий расстояний из CSV
CATEGORY_FILE = 'category.csv'

def load_distance_categories(file_path):
    """Загружает категории расстояний из CSV файла"""
    categories = []
    with open(file_path, 'r', encoding='utf-8') as f:
        # Пропускаем заголовок, шляпа нужна!!!
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 3:
                min_val = float(parts[0])
                if parts[1] == 'inf':
                    max_val = float('inf')
                else:
                    max_val = float(parts[1])
                category = parts[2]
                categories.append({
                    'min': min_val,
                    'max': max_val,
                    'category': category
                })
    return categories

# Загружаем категории
distance_categories = load_distance_categories(CATEGORY_FILE)

def get_distance_category(distance):
    """Определяет категорию для заданного расстояния"""
    for cat_dis in distance_categories:
        if cat_dis['min'] <= distance <= cat_dis['max']:
            return cat_dis['category']
    return "Не определено"

timestamp_file = os.path.join(DATA_DIR, 'timestamp.txt')

with open(timestamp_file, 'r', encoding='utf-8') as file:
    TIMESTAMP = file.readlines()[-1].strip()

# путь к файлам по последней дате
RUN_DIR = os.path.join(DATA_DIR, TIMESTAMP)
# папка для сохранения результатов
os.makedirs(OUTPUT_DIR_USER, exist_ok=True)

# Загружаем scaler
with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Загружаем DataFrame специалистов (для Id)
with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_df_spec.pkl'), 'rb') as f:
    df_spec = pickle.load(f)

# Загружаем масштабированные данные специалистов
X_spec_scaled = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_X_scaled.npy'))

# Загружаем профессии специалистов
professions = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_professions.npy'), allow_pickle=True)

# Загружаем список признаков
feature_cols = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_feature_cols.npy'), 
                       allow_pickle=True)

# Получаем Id специалистов
spec_ids = df_spec['Id'].values

# Определяем доступные K
k_values = []
for file in os.listdir(RUN_DIR):
    if file.startswith(f'{OUTPUT_PREFIX}_labels_k') and file.endswith('.npy'):
        k = int(file.split('_k')[-1].split('.')[0])
        k_values.append(k)
k_values = sorted(k_values)

# Загружаем модели KMeans и метки кластеров
kmeans_models = {}
cluster_labels_spec = {}

for k in k_values:
    # Загружаем модель
    with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_kmeans_k{k}.pkl'), 'rb') as f:
        kmeans_models[k] = pickle.load(f)
    
    # Загружаем метки кластеров
    cluster_labels_spec[k] = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_labels_k{k}.npy'))

# Загружаем данные пользователей
df_users = get_parsing.parsing("Pupil")

# Берем только те колонки, которые есть в feature_cols
available_feature_cols = [col for col in feature_cols if col in df_users.columns]

# масштабирование
X_users = df_users[available_feature_cols].values
X_users_scaled = scaler.transform(X_users)

# Получаем ID и ФИО пользователей
user_ids = df_users['Id'].values
user_names = df_users['fullName'].values if 'fullName' in df_users.columns else [''] * len(df_users)

# ----------------------- АНАЛИЗ ПОЛЬЗОВАТЕЛЕЙ -----------------------
results_rows = []

for user_idx, (user_id, user_name, user_vec) in enumerate(zip(user_ids, 
                                                              user_names, X_users_scaled)):
    row = {
        'user_id': int(user_id),
        'user_name': str(user_name)  # временно добавили ФИО или не временно? 
    }

    for k in k_values:
        kmeans = kmeans_models[k]
        labels = cluster_labels_spec[k]

        user_cluster = kmeans.predict([user_vec])[0]
        indices_in_cluster = np.where(labels == user_cluster)[0]

        distances = cdist([user_vec], X_spec_scaled[indices_in_cluster])[0]
        
        # Сортируем по расстоянию
        sorted_indices = np.argsort(distances)
        
        # Ближайший специалист
        nearest_idx_in_cluster = np.argmin(distances)
        nearest_spec_position = indices_in_cluster[nearest_idx_in_cluster]
        nearest_distance = distances[nearest_idx_in_cluster]
        predicted_profession = professions[nearest_spec_position]
        nearest_spec_id = spec_ids[nearest_spec_position]
        
        # Определяем категорию для ближайшего расстояния
        distance_category = get_distance_category(nearest_distance)

        row[f'K{k}_cluster'] = int(user_cluster)
        row[f'K{k}_nearest_spec_id'] = int(nearest_spec_id)
        row[f'K{k}_distance'] = float(nearest_distance)
        row[f'K{k}_distance_category'] = distance_category
        row[f'K{k}_profession'] = str(predicted_profession)
        
        # Для каждой категории находим ВСЕХ подходящих специалистов
        categories_specialists = {}
        
        for cat in distance_categories:
            cat_name = cat['category']
            cat_min = cat['min']
            cat_max = cat['max']
            
            # Находим всех специалистов, попадающих в этот диапазон
            matching_specialists = []
            for idx in sorted_indices:
                spec_pos = indices_in_cluster[idx]
                dist = distances[idx]
                
                if cat_min <= dist <= cat_max:
                    matching_specialists.append({
                        'spec_id': int(spec_ids[spec_pos]),
                        'profession': str(professions[spec_pos]),
                        'distance': float(dist)
                    })
            
            categories_specialists[cat_name] = matching_specialists
        
        row[f'K{k}_categories'] = categories_specialists

    results_rows.append(row)

# ----------------------- СОХРАНЕНИЕ -----------------------
for row in results_rows:
    user_id = row['user_id']
    user_file = os.path.join(OUTPUT_DIR_USER, f'user_id_{user_id}.json')
    with open(user_file, 'w', encoding='utf-8') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)

# все пользователи, скорее всего уберем
all_users_file = os.path.join(OUTPUT_DIR_USER, 'all_users.json')
with open(all_users_file, 'w', encoding='utf-8') as f:
    json.dump(results_rows, f, ensure_ascii=False, indent=2)