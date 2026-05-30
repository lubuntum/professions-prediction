import requests

def start():

    url = "https://api.evcg.ru/api/auth/login"
    data = {
        "email": "lubuntum@gmail.com",  
        "password": "123456789"         
    }

    response = requests.post(url, json=data)
    return response.text

# print(start())