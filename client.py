# client.py
import requests


body = {
"team": "Manchester Utd",
"season_end_year": 2025
}


resp = requests.post(url='http://127.0.0.1:8000/score', json=body)
print(resp.status_code)
print(resp.json())
