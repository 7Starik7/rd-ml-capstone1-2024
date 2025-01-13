import requests

url = 'http://localhost:9696/predict'

# data = {'url': 'http://bit.ly/mlbookcamp-pants'}
data = {'url': 'https://c7.alamy.com/comp/K0W4HC/a-traffic-sign-indicating-a-speed-limit-of-50-kmh-seen-near-tubingen-K0W4HC.jpg'}

result = requests.post(url, json=data).json()
print(result)