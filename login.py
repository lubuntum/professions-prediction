import requests

def start():
    """
    Тут будет документация или нет
    """

    url = "https://api.evcg.ru/api/auth/login"
    data = {
        "email": "lubuntum@gmail.com",  
        "password": "123456789"         
    }

    response = requests.post(url, json=data, timeout=10)
    return response.text

# print(start())
