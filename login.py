import requests

url = "https://api.evcg.ru/api/auth/login"
data = {
    "email": "lubuntum@gmail.com",  
    "password": "123456789"         
}

response = requests.post(url, json=data)
print("Статус:", response.status_code)
print("Токен:", response.text)