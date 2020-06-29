import requests 
from input_data import input

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": input}

r = requests.get(URL,headers=headers, json=data) 

r.json()