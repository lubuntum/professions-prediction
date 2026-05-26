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

# ======================== НАСТРОЙКИ ========================
# Дата и время
DATA_DIR = 'data'  # Папка для данных
OUTPUT_PREFIX = 'clustering_results'
OUTPUT_DIR_USER = "data_user"  # Папка для результатов учеников

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

# Загружаем масштабированные данные специалистов
X_spec_scaled = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_X_scaled.npy'))

# Загружаем профессии специалистов
professions = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_professions.npy'), allow_pickle=True)

# Загружаем список признаков
feature_cols = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_feature_cols.npy'), allow_pickle=True)

# Определяем доступные K (по наличию файлов labels)
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


"""
# Файл с данными пользователей
USER_FILE = 'ВМИП_user_data_fin_03.05.2026_test.csv'

# Колонки для исключения (те же, что и в первой части)
COLS_TO_EXCLUDE_SPEC = ['Период работы по профессии', 'Степень привлекательности профессии']

# Колонки для исключения из данных пользователей
COLS_TO_EXCLUDE_USER = ['Период работы по профессии', 'Степень привлекательности профессии', 'класс']

PROF_COL = 'Группа профессий'  # Название колонки с профессиями

# Папка для сохранения результатов второй части
OUTPUT_DIR = "data_user"
# ===========================================================

# Создаём папку для результатов
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Временная метка для файлов второй части
run_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

print("="*70)
print("ЧАСТЬ 2: ОПРЕДЕЛЕНИЕ ПРОФЕССИИ ПОЛЬЗОВАТЕЛЕЙ")
print("="*70)
print(f"Загрузка данных из папки: {DATA_DIR}")
print(f"Timestamp из первой части: {TIMESTAMP}")
print(f"Файл пользователей: {USER_FILE}\n")
print(f"Результаты будут сохранены в папку: {OUTPUT_DIR}\n")

# ----------------------- 1. ЗАГРУЗКА РЕЗУЛЬТАТОВ ПЕРВОЙ ЧАСТИ -----------------------
print("="*60)
print("1. ЗАГРУЗКА РЕЗУЛЬТАТОВ КЛАСТЕРИЗАЦИИ (ЧАСТЬ 1)")
print("="*60)

# Загружаем scaler
scaler_path = os.path.join(DATA_DIR, f'clustering_results_scaler_{TIMESTAMP}.pkl')
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"✅ Загружен scaler: {scaler_path}")
print(f"   Ожидаемое количество признаков: {scaler.n_features_in_}")

# Загружаем данные специалистов (масштабированные)
X_scaled_path = os.path.join(DATA_DIR, f'clustering_results_X_scaled_{TIMESTAMP}.npy')
X_spec_scaled = np.load(X_scaled_path)
print(f"✅ Загружены масштабированные данные специалистов: {X_spec_scaled.shape}")

# Загружаем профессии специалистов
professions_path = os.path.join(DATA_DIR, f'clustering_results_professions_{TIMESTAMP}.npy')
professions = np.load(professions_path, allow_pickle=True)
print(f"✅ Загружены профессии специалистов: {len(professions)}")

# Загружаем уникальные профессии
unique_professions = np.unique(professions)
n_professions = len(unique_professions)
print(f"✅ Уникальных профессий: {n_professions}")

# Загружаем названия признаков из файла специалистов

if os.path.exists(SPECIALIST_FILE):
    df_spec_temp = pd.read_csv(SPECIALIST_FILE, encoding='utf-8')
    df_spec_clean_temp = df_spec_temp.drop(columns=COLS_TO_EXCLUDE_SPEC)
    feature_cols_from_spec = [col for col in df_spec_clean_temp.columns if col != PROF_COL]
    print(f"✅ Загружены названия признаков из файла специалистов: {len(feature_cols_from_spec)} признаков")
else:
    feature_cols_from_spec = None

# Определяем доступные K
k_values = list(range(2, n_professions + 1))
print(f"✅ Доступные K для анализа: {k_values}")

# Загружаем модели KMeans для каждого K
kmeans_models = {}
for k in k_values:
    model_path = os.path.join(DATA_DIR, f'clustering_results_kmeans_k{k}_{TIMESTAMP}.pkl')
    with open(model_path, 'rb') as f:
        kmeans_models[k] = pickle.load(f)
print(f"✅ Загружены модели KMeans для K = {min(k_values)}..{max(k_values)}")

# Загружаем метки кластеров для специалистов
cluster_labels_spec = {}
for k in k_values:
    labels_path = os.path.join(DATA_DIR, f'clustering_results_labels_k{k}_{TIMESTAMP}.npy')
    cluster_labels_spec[k] = np.load(labels_path)
print(f"✅ Загружены метки кластеров специалистов для каждого K")

print("\n")

# ----------------------- 2. ЗАГРУЗКА ДАННЫХ ПОЛЬЗОВАТЕЛЕЙ -----------------------
print("="*60)
print("2. ЗАГРУЗКА ДАННЫХ ПОЛЬЗОВАТЕЛЕЙ")
print("="*60)

if not os.path.exists(USER_FILE):
    raise FileNotFoundError(f"❌ Файл {USER_FILE} не найден!")

df_users = pd.read_csv(USER_FILE, encoding='utf-8')
print(f"✅ Загружены пользователи: {df_users.shape[0]} строк, {df_users.shape[1]} столбцов")

# Исключаем служебные колонки
if feature_cols_from_spec is not None:
    available_cols = [col for col in feature_cols_from_spec if col in df_users.columns]
    missing_cols = [col for col in feature_cols_from_spec if col not in df_users.columns]
    
    if missing_cols:
        print(f"⚠️ Отсутствуют признаки у пользователей: {missing_cols}")
    
    user_feature_cols = available_cols
else:
    user_feature_cols = [col for col in df_users.columns 
                         if col not in COLS_TO_EXCLUDE_USER + [PROF_COL, 'ФИО', 'ID', 'id']
                         and pd.api.types.is_numeric_dtype(df_users[col])]

print(f"✅ Итоговые признаки для пользователей: {len(user_feature_cols)}")

# Проверяем соответствие количества признаков
expected_n_features = scaler.n_features_in_
if len(user_feature_cols) != expected_n_features:
    print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: Количество признаков не совпадает!")
    print(f"   scaler ожидает {expected_n_features} признаков")
    print(f"   У пользователей найдено {len(user_feature_cols)} признаков")
    raise ValueError("Несоответствие количества признаков")

# Извлекаем матрицу признаков пользователей
X_users = df_users[user_feature_cols].values
X_users_scaled = scaler.transform(X_users)
print(f"✅ Данные пользователей масштабированы: {X_users_scaled.shape}")

# Получаем список ФИО пользователей
if 'ФИО' in df_users.columns:
    user_names = df_users['ФИО'].values
else:
    user_names = [f"Пользователь_{i}" for i in range(len(df_users))]

print(f"✅ Найдено пользователей: {len(user_names)}")
print("\n")

# ----------------------- 3. АНАЛИЗ ПОЛЬЗОВАТЕЛЕЙ -----------------------
print("="*60)
print("3. ОПРЕДЕЛЕНИЕ ПРОФЕССИЙ ПОЛЬЗОВАТЕЛЕЙ")
print("="*60)

# Структура для хранения результатов (одна строка на пользователя)
all_results_rows = []

# Для каждого пользователя
for user_idx, (user_name, user_vec) in enumerate(zip(user_names, X_users_scaled)):
    print(f"\n--- Пользователь {user_idx+1}/{len(user_names)}: {user_name} ---")
    
    # Строка результатов для этого пользователя
    row = {
        'user_name': user_name,
        'user_index': user_idx
    }
    
    # Для каждого K
    for k in k_values:
        # Получаем модель KMeans для текущего K
        kmeans = kmeans_models[k]
        
        # Определяем кластер пользователя
        user_cluster = kmeans.predict([user_vec])[0]
        
        # Находим всех специалистов в этом кластере
        mask_spec_in_cluster = cluster_labels_spec[k] == user_cluster
        indices_in_cluster = np.where(mask_spec_in_cluster)[0]
        
        if len(indices_in_cluster) > 0:
            # Вычисляем расстояния от пользователя до каждого специалиста в кластере
            distances = cdist([user_vec], X_spec_scaled[indices_in_cluster])[0]
            
            # Находим ближайшего специалиста
            nearest_idx_in_cluster = np.argmin(distances)
            nearest_spec_idx = indices_in_cluster[nearest_idx_in_cluster]
            nearest_distance = distances[nearest_idx_in_cluster]
            
            # Получаем профессию ближайшего специалиста
            predicted_profession = professions[nearest_spec_idx]
        else:
            nearest_spec_idx = -1
            nearest_distance = np.inf
            predicted_profession = "Нет специалистов в кластере"
        
        # Добавляем результаты для этого K в строку
        row[f'K{k}'] = k
        row[f'K{k}_cluster'] = int(user_cluster)
        row[f'K{k}_nearest_idx'] = int(nearest_spec_idx)
        row[f'K{k}_distance'] = float(nearest_distance)
        row[f'K{k}_profession'] = predicted_profession
        
        # Краткий вывод
        prof_short = predicted_profession[:30] + "..." if len(predicted_profession) > 30 else predicted_profession
        print(f"  K={k:2d}: кластер {user_cluster} -> {prof_short} (расст. {nearest_distance:.4f})")
    
    all_results_rows.append(row)

print("\n")

# ----------------------- 4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ -----------------------
print("="*60)
print("4. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("="*60)

# Создаём DataFrame с одной строкой на пользователя
df_results = pd.DataFrame(all_results_rows)

# Сохраняем CSV
csv_filename = f'user_predictions_{run_timestamp}.csv'
csv_path = os.path.join(OUTPUT_DIR, csv_filename)
df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"✅ Сохранен CSV: {csv_path}")
print(f"   {len(df_results)} строк (пользователей)")
print(f"   {len(df_results.columns)} столбцов")

# Сохраняем JSON
json_filename = f'user_predictions_{run_timestamp}.json'
json_path = os.path.join(OUTPUT_DIR, json_filename)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_results_rows, f, ensure_ascii=False, indent=2, default=str)
print(f"✅ Сохранен JSON: {json_path}")

# ----------------------- 5. ВЫВОД СТРУКТУРЫ ФАЙЛА -----------------------
print("\n" + "="*60)
print("АНАЛИЗ ЗАВЕРШЕН")
print("="*60)
print(f"📁 Результаты сохранены в папку: {OUTPUT_DIR}")
print(f"   • {csv_filename} - одна строка на пользователя")
print(f"   • {json_filename} - структурированные результаты")

"""