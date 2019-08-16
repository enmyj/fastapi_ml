#%%
import requests
import json

url = 'http://127.0.0.1:8000/titanic/'
params = {"Age": 18, "Sex": 'female', "Embarked": 'S'}
# params = [{"Age":18,"Sex":"female","Embarked":"S"},{"Age":21,"Sex":"male","Embarked":"S"}]

# response = requests.get(url, params)
response = requests.post(url, data = json.dumps(params))

print(response.json())


#%%
