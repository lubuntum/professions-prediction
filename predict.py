import requests
import json
import os
import login

def create_prediction():
    """
    Отправляет JSON файл на API для создания предсказания
    """
    url = "https://api.evcg.ru/api/predictions/create"
    
    token = login.start()
    
    headers = {
        "Authorization": token
    }
    
    # Данные для предсказания
    prediction_data = {
        "pupilId": 211,  # из вашего JSON файла
        "predictionType": "CLUSTER"  # или "MATH"
    }
    
    # Файл для загрузки
    file_path = "user_id_211.json"
    
    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден!")
        return None
    
    # Отправляем запрос
    with open(file_path, 'rb') as f:
        files = {
            "prediction": (
                "",
                json.dumps(prediction_data),
                "application/json"
            ),
            "file": (
                file_path,
                f,
                "application/json"
            )
        }
        
        response = requests.post(url, headers=headers, files=files)
    
    # Выводим результат
    print(f"Статус: {response.status_code}")
    
    try:
        result = response.json()
        print("Ответ:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except:
        print("Ответ (не JSON):")
        print(response.text)
        return response.text
create_prediction()