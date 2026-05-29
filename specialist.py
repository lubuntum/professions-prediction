"""
Часть 1: Кластеризация данных специалистов
===========================================
Назначение:
    1. Загрузить и подготовить данные о специалистах
    2. Выполнить кластеризацию методом K-means для K = 2..N (N = количество уникальных профессий)
    3. Сохранить результаты: центроиды кластеров, метки кластеров для каждого специалиста,
       PCA-модель и масштабатор для учеников
"""
import pickle
import warnings
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import get_parsing

# чтобы в консоли ничего не было 
warnings.filterwarnings('ignore')

# ======================== НАСТРОЙКИ ========================
df_spec = get_parsing.parsing("Specialist")  # Файл с данными специалистов (сервер)
PROF_COL = 'profession'  # Название колонки с профессиями
OUTPUT_PREFIX = 'clustering_results'  # Префикс для выходных файлов
OUTPUT_DIR = 'data'  # Папка для сохранения результатов

# Создаем папку data, если её нет
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Получаем текущую дату и время для имени файлов
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

# Сохраняем timestamp в файл (в папке data)
timestamp_file = os.path.join(OUTPUT_DIR, 'timestamp.txt')
with open(timestamp_file, 'a', encoding='utf-8') as f:
    f.write(timestamp + '\n')

# создается папку для запуска
run_dir = os.path.join(OUTPUT_DIR, timestamp)
os.makedirs(run_dir, exist_ok=True)

# Сохраняем исходный DataFrame (внутри папки с timestamp)
DF_SPEC_FILENAME = f'{OUTPUT_PREFIX}_df_spec.pkl'
df_spec_path = os.path.join(run_dir, DF_SPEC_FILENAME)
with open(df_spec_path, 'wb') as f:
    pickle.dump(df_spec, f)

# Определяем признаки и целевую переменную (профессию)
# убираем Id и имя
feature_cols = [col for col in df_spec.columns
                if col not in [PROF_COL, 'Id', 'fullName']]
X = df_spec[feature_cols].values
professions = df_spec[PROF_COL].values
unique_professions = np.unique(professions)
n_professions = len(unique_professions)

# потом убрать!!!
# print(f"Уникальных профессий: {n_professions}")
# print(f"Профессии: {unique_professions}")
# print(len(df_spec))


# ----------------------- МАСШТАБИРОВАНИЕ -----------------------
# Масштабируем данные (StandardScaler: вычитаем среднее, делим на std)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Сохраняем scaler для использования в учениках
SCALER_FILENAME = f'{OUTPUT_PREFIX}_scaler.pkl'
scaler_path = os.path.join(run_dir, SCALER_FILENAME)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# ----------------------- КЛАСТЕРИЗАЦИЯ -----------------------

# K от 2 до количества профессий
for k in range(2, n_professions + 1):  
    # Обучаем K-means
    # n_init=10 использовали, увеличу до 50 для точности, будет грузить систему
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
    cluster_labels = kmeans.fit_predict(X_scaled)
       
    # Сохраняем модель для текущего K
    model_filename = f'{OUTPUT_PREFIX}_kmeans_k{k}.pkl'
    model_path = os.path.join(run_dir, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump(kmeans, f)

# ----------------------- PCA ПРЕОБРАЗОВАНИЕ -----------------------
# Обучаем PCA для 2D и 3D визуализации (используем данные после кластеризации)
# PCA будет нужен для учеников

# PCA для 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

# PCA для 3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Сохраняем PCA модели
PCA_2D_FILENAME = f'{OUTPUT_PREFIX}_pca_2d.pkl'
pca_2d_path = os.path.join(run_dir, PCA_2D_FILENAME)
with open(pca_2d_path, 'wb') as f:
    pickle.dump(pca_2d, f)

PCA_3D_FILENAME = f'{OUTPUT_PREFIX}_pca_3d.pkl'
pca_3d_path = os.path.join(run_dir, PCA_3D_FILENAME)
with open(pca_3d_path, 'wb') as f:
    pickle.dump(pca_3d, f)

# ----------------------- СОХРАНЕНИЕ ДАННЫХ ДЛЯ Учеников -----------------------

# 1. Профессии специалистов
np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_professions.npy'), professions)

# 2. Масштабированные данные
np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_X_scaled.npy'), X_scaled)

# 3. Метки кластеров для каждого K
for k in range(2, n_professions + 1):
    model_path = os.path.join(run_dir, f'{OUTPUT_PREFIX}_kmeans_k{k}.pkl')
    with open(model_path, 'rb') as f:
        kmeans = pickle.load(f)
    np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_labels_k{k}.npy'), kmeans.labels_)

# 4. PCA данные
np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_X_pca_2d.npy'), X_pca_2d)
np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_X_pca_3d.npy'), X_pca_3d)

# 5. Признаки
np.save(os.path.join(run_dir, f'{OUTPUT_PREFIX}_feature_cols.npy'), feature_cols)
