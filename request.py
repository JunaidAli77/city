import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Year':2017, 'City':101})

print(r.json())