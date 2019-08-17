#%%
import requests
import json

url = 'http://127.0.0.1:8000/titanic/'
params = {"age": 18, "sex": "female", "embarked": "S"}
# params = [{"age":18,"sex":"female","embarked":"S"},{"age":21,"sex":"male","embarked":"S"}]

response = requests.post(url, json = params)

print(response.json())


#%%
