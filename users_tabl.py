"""
Анализ профессий для каждого пользователя
===========================================
Назначение:
    1. Для каждого JSON-файла пользователя из папки data_user
    2. Найти все K2_categories, K3_categories, K4_categories и т.д.
    3. Извлечь profession и distance
    4. Вывести для каждого пользователя:
       - Уникальные профессии
       - Количество встречаемости
       - Среднее арифметическое distance
       - Медиана distance
       - Стандартное отклонение distance
       - Среднее геометрическое distance
"""

import os
import json
import glob
import numpy as np

# ======================== НАСТРОЙКИ ========================
DATA_USER_DIR = "data_user"

def geometric_mean(values):
    """Вычисляет среднее геометрическое списка значений"""
    if not values:
        return None
    values = [float(v) for v in values if v > 0]
    if not values:
        return None
    product = np.prod(values)
    return product ** (1.0 / len(values))

def analyze_user_file(file_path):
    """Анализирует один JSON-файл пользователя и выводит результаты"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    user_id = data.get('user_id', 'Unknown')
    user_name = data.get('user_name', 'Unknown')
    
    print("\n" + "=" * 110)
    print(f"👤 Пользователь: {user_name} (ID: {user_id})")
    print("=" * 110)
    
    # Собираем все profession и distance из всех категорий
    all_professions = []
    
    for key, value in data.items():
        if key.endswith('_categories') and key.startswith('K'):
            for category, specialists in value.items():
                if not specialists:
                    continue
                
                for specialist in specialists:
                    profession = specialist.get('profession', 'Неизвестно')
                    distance = specialist.get('distance', 0)
                    all_professions.append({
                        'profession': profession,
                        'distance': distance
                    })
    
    if not all_professions:
        print("  Нет данных о специалистах")
        return
    
    # Группируем по профессиям
    profession_stats = {}
    
    for item in all_professions:
        profession = item['profession']
        distance = item['distance']
        
        if profession not in profession_stats:
            profession_stats[profession] = {
                'distances': [],
                'count': 0
            }
        
        profession_stats[profession]['distances'].append(distance)
        profession_stats[profession]['count'] += 1
    
    # Выводим результаты для пользователя
    print(f"\n{'Профессия':<45} {'Встреч.':<8} {'Ср.арифм.':<12} {'Медиана':<12} {'Ст.откл.':<12} {'Ср.геом.':<12}")
    print("-" * 110)
    
    # Сортируем по встречаемости
    sorted_professions = sorted(profession_stats.items(), 
                               key=lambda x: x[1]['count'], 
                               reverse=True)
    
    for profession, stats in sorted_professions:
        distances = stats['distances']
        arithmetic_mean = round(np.mean(distances), 4)
        median = round(np.median(distances), 4)
        std = round(np.std(distances), 4)
        geom_mean = round(geometric_mean(distances), 4)
        
        # Обрезаем длинные названия профессий
        prof_name = profession[:42] + "..." if len(profession) > 45 else profession
        
        print(f"{prof_name:<45} {stats['count']:<8} {arithmetic_mean:<12} {median:<12} {std:<12} {geom_mean:<12}")
    
    print("-" * 110)
    print(f"📈 Всего уникальных профессий: {len(profession_stats)}")
    print(f"📊 Всего записей: {len(all_professions)}")

def main():
    json_files = glob.glob(os.path.join(DATA_USER_DIR, "user_id_*.json"))
    
    if not json_files:
        print(f"Файлы не найдены в папке {DATA_USER_DIR}")
        return
    
    print(f"Найдено {len(json_files)} файлов пользователей")
    
    for file_path in sorted(json_files):
        analyze_user_file(file_path)
    
    print("\n" + "=" * 110)
    print("✅ Анализ завершён")
    print("=" * 110)

if __name__ == "__main__":
    main()