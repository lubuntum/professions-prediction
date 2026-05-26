import requests

def start():

    url = "https://api.evcg.ru/api/auth/login"
    data = {
        "email": "lubuntum@gmail.com",  
        "password": "123456789"         
    }

    response = requests.post(url, json=data)
    with open('token.txt', 'w', encoding="utf-8") as file:
        file.write(response.text)