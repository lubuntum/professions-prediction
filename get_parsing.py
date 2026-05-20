from datetime import datetime
import requests
import json


def parsing(parse_type="All"):
    """
    Данная функция будет возвращать пользователей с сайта
    По умолчанию type="All" - все пользователи
    Pupil - ученики
    Specialist - специалисты 
    """

    # Текущая дата и время, текущая
    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%dT%H:%M:%S")

    # Токен
    with open('token.txt', 'r', encoding="utf-8") as file:
        token = file.read()

    url = "https://api.evcg.ru/api/psych-tests/completed-tests"
    params = {
        "type": parse_type,
        "startDate": "2026-04-01T00:00:00",
        "endDate": formatted_date
    }
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        print(f"Найдено пользователей: {len(data)}")
    except requests.exceptions.ReadTimeout:
        return None
    
    for fio in data:
        print(fio.get('accountId'), fio.get('fullName'), len(fio.get('psychTests', [])))

    
    # if data:
    #     first = data[223]
    #     print(f"\nПервый пользователь:")
    #     print(f"  акаунт: {first.get('accountId')}")
    #     print(f"  ФИО: {first.get('fullName')}")
    #     print(f"  Email: {first.get('email')}")
    #     print(f"  Тестов пройдено: {len(first.get('psychTests', []))}")
        
    #     # Какие тесты проходил
    #     for test in first.get('psychTests', []):
    #         print(f"    - {test.get('testTypeName')}: {test.get('createdAt')[:10]}")

parsing("Specialist")
