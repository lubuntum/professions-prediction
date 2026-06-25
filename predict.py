import requests
import json
import os
import login
import time

def create_prediction():
    """
    Отправляет все JSON файлы из папки data_user на API для создания предсказаний
    """
    url = "https://api.evcg.ru/api/predictions/create"

    token = login.start()

    headers = {
        "Authorization": token
    }

    # Папка с JSON файлами пользователей
    data_user_dir = "data_user"

    # Получаем все JSON файлы в папке
    json_files = [f for f in os.listdir(data_user_dir) if f.startswith('user_id_') and f.endswith('.json')]

    print(f"Найдено файлов: {len(json_files)}")

    for file_name in json_files:
        file_path = os.path.join(data_user_dir, file_name)

        # Извлекаем pupilId из имени файла
        pupil_id = int(file_name.split('_')[-1].split('.')[0])

        print(f"\n{'='*50}")
        print(f"Обработка: {file_name}")
        print(f"Pupil ID: {pupil_id}")

        # Данные для предсказания
        prediction_data = {
            "pupilId": pupil_id,
            "predictionType": "CLUSTER"
        }

        # Отправляем запрос
        with open(file_path, 'rb') as f:
            files = {
                "prediction": (
                    "",
                    json.dumps(prediction_data),
                    "application/json"
                ),
                "file": (
                    file_name,
                    f,
                    "application/json"
                )
            }

            response = requests.post(url, headers=headers, files=files)

        print(f"Статус: {response.status_code}")

        try:
            result = response.json()
            print("Ответ:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except:
            print("Ответ (не JSON):")
            print(response.text)

        # Задержка 2 секунда между запросами
        # time.sleep(2)

create_prediction()
