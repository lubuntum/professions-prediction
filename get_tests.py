import requests

token = "скопируй сюда свой токен"  # тот, что получил на шаге 2

url = "https://api.evcg.ru/api/psych-tests/completed-tests"
params = {
    "type": "Pupil",
    "startDate": "2026-04-01T00:00:00",
    "endDate": "2026-04-30T23:59:59"
}
headers = {
    "Authorization": f"Bearer {token}"
}

response = requests.get(url, params=params, headers=headers)
print("Статус:", response.status_code)
print("Данные:", response.text)