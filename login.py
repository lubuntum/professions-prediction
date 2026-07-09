import os
import requests
from dotenv import load_dotenv

load_dotenv()

def start():
    url = f"{os.getenv('API_BASE_URL')}/api/auth/login"
    data = {
        "email": os.getenv("API_EMAIL"),
        "password": os.getenv("API_PASSWORD")
    }
    response = requests.post(url, json=data, timeout=10)
    return response.text

# print(start())
