"""
Часть 3: Визуализация результатов кластеризации
=================================================
Назначение:
    1. Загрузить результаты кластеризации из первой части
    2. Построить 2D и 3D PCA графики для всех K
    3. Наложение пользователей на графики специалистов
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

import get_parsing

# ======================== НАСТРОЙКИ ========================
DATA_DIR = 'data'
OUTPUT_PREFIX = 'clustering_results'
OUTPUT_VIZ_DIR = 'visualizations'

# Создаем папку для визуализаций
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)

# Определяем последний timestamp
timestamp_file = os.path.join(DATA_DIR, 'timestamp.txt')
with open(timestamp_file, 'r', encoding='utf-8') as file:
    TIMESTAMP = file.readlines()[-1].strip()

RUN_DIR = os.path.join(DATA_DIR, TIMESTAMP)


# ----------------------- ЗАГРУЗКА ДАННЫХ -----------------------
# Загружаем DataFrame специалистов
with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_df_spec.pkl'), 'rb') as f:
    df_spec = pickle.load(f)

# Загружаем PCA модели
with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_pca_2d.pkl'), 'rb') as f:
    pca_2d = pickle.load(f)

with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_pca_3d.pkl'), 'rb') as f:
    pca_3d = pickle.load(f)

# Загружаем масштабированные данные
X_spec_scaled = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_X_scaled.npy'))

# Загружаем список признаков
feature_cols = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_feature_cols.npy'), allow_pickle=True)

# Определяем все доступные K
k_values = []
for file in os.listdir(RUN_DIR):
    if file.startswith(f'{OUTPUT_PREFIX}_labels_k') and file.endswith('.npy'):
        k = int(file.split('_k')[-1].split('.')[0])
        k_values.append(k)
k_values = sorted(k_values)

# Загружаем профессии
professions = df_spec['profession'].values

# ----------------------- PCA ПРЕОБРАЗОВАНИЕ -----------------------
X_spec_pca_2d = pca_2d.transform(X_spec_scaled)
X_spec_pca_3d = pca_3d.transform(X_spec_scaled)

# ----------------------- ЗАГРУЗКА ПОЛЬЗОВАТЕЛЕЙ -----------------------
# Загружаем данные пользователей
df_users = get_parsing.parsing("Pupil")

# Загружаем scaler
with open(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Подготавливаем данные пользователей
available_feature_cols = [col for col in feature_cols if col in df_users.columns]
X_users = df_users[available_feature_cols].values
X_users_scaled = scaler.transform(X_users)

# Трансформируем пользователей в PCA пространство
X_users_pca_2d = pca_2d.transform(X_users_scaled)
X_users_pca_3d = pca_3d.transform(X_users_scaled)

user_ids = df_users['Id'].values
user_names = df_users['fullName'].values if 'fullName' in df_users.columns else [f'User_{i}' for i in range(len(df_users))]

# ----------------------- ПОСТРОЕНИЕ ГРАФИКОВ ДЛЯ ВСЕХ K -----------------------


cmap = plt.cm.tab10

for k in k_values:  
    # Загружаем метки для текущего K
    cluster_labels = np.load(os.path.join(RUN_DIR, f'{OUTPUT_PREFIX}_labels_k{k}.npy'))
    unique_clusters = np.unique(cluster_labels)
    
    # ==================== 2D ГРАФИК СПЕЦИАЛИСТОВ ====================
    fig_2d, ax_2d = plt.subplots(figsize=(14, 10))
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        color = cmap(cluster_id % 10)
        ax_2d.scatter(X_spec_pca_2d[mask, 0], X_spec_pca_2d[mask, 1],
                      c=[color], s=50, alpha=0.6, edgecolors='black', linewidth=0.5,
                      label=f'Кластер {cluster_id}')
    
    ax_2d.set_xlabel('Первая главная компонента', fontsize=12)
    ax_2d.set_ylabel('Вторая главная компонента', fontsize=12)
    ax_2d.set_title(f'PCA визуализация специалистов (K={k})', fontsize=14)
    ax_2d.legend(loc='best', fontsize=10)
    ax_2d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIZ_DIR, f'spec_clusters_2d_K{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 3D ГРАФИК СПЕЦИАЛИСТОВ ====================
    fig_3d = plt.figure(figsize=(14, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        color = cmap(cluster_id % 10)
        ax_3d.scatter(X_spec_pca_3d[mask, 0], X_spec_pca_3d[mask, 1], X_spec_pca_3d[mask, 2],
                      c=[color], s=40, alpha=0.5, edgecolors='black', linewidth=0.3,
                      label=f'Кластер {cluster_id}')
    
    ax_3d.set_xlabel('Первая главная компонента', fontsize=10)
    ax_3d.set_ylabel('Вторая главная компонента', fontsize=10)
    ax_3d.set_zlabel('Третья главная компонента', fontsize=10)
    ax_3d.set_title(f'3D PCA визуализация специалистов (K={k})', fontsize=12)
    ax_3d.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIZ_DIR, f'spec_clusters_3d_K{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # ==================== 2D ГРАФИК С ПОЛЬЗОВАТЕЛЯМИ ====================
    fig_users, ax_users = plt.subplots(figsize=(14, 10))
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        color = cmap(cluster_id % 10)
        ax_users.scatter(X_spec_pca_2d[mask, 0], X_spec_pca_2d[mask, 1],
                         c=[color], s=50, alpha=0.5, edgecolors='black', linewidth=0.3,
                         label=f'Кластер {cluster_id} (специалисты)')
    
    # Пользователи - красными крестиками
    ax_users.scatter(X_users_pca_2d[:, 0], X_users_pca_2d[:, 1],
                     c='red', s=80, marker='x', linewidths=2, label='Пользователи')
    
    ax_users.set_xlabel('Первая главная компонента', fontsize=12)
    ax_users.set_ylabel('Вторая главная компонента', fontsize=12)
    ax_users.set_title(f'PCA визуализация специалистов и пользователей (K={k})', fontsize=14)
    ax_users.legend(loc='best', fontsize=10)
    ax_users.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_VIZ_DIR, f'spec_and_users_2d_K{k}.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ----------------------- ИНФОРМАЦИЯ О ЗАПУСКЕ -----------------------
info = {
    'timestamp': TIMESTAMP,
    'k_values': k_values,
    'n_specialists': len(df_spec),
    'n_users': len(df_users),
    'n_features': len(feature_cols)
}

with open(os.path.join(OUTPUT_VIZ_DIR, 'viz_info.json'), 'w', encoding='utf-8') as f:
    json.dump(info, f, ensure_ascii=False, indent=2)
