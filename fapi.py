from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from joblib import load
import pandas as pd
import json

def preprocess(df):
    df = df.dropna(axis=0)
    # process into proper format
    df_ = pd.get_dummies(df)
    return df_.reindex(columns = model_columns, fill_value = 0)


#%% initialize fast api instance

app = FastAPI()

#%% load models
model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    model = load(f)

model_columns_path = 'model_columns.pkl'
with open(model_columns_path, 'rb') as f:
    model_columns = load(f)

#%% data model?

class TitanicData(BaseModel):
    age: int
    sex: str
    embarked: str



#%%
@app.post("/titanic/")
async def titanic(data: TitanicData):

    df = pd.DataFrame([dict(data)])

    query = preprocess(df)

    # make prediction using preprocessed data
    prediction = list(model.predict(query))

    # create JSON object and return
    return json.dumps({'prediction': str(prediction)})

@app.post('/titanic_upload/')
async def titanic_upload(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df = df[['Age','Sex','Embarked']]

    query = preprocess(df)

    # make prediction using preprocessed data
    prediction = list(model.predict(query))

    # create JSON object and return
    return json.dumps({'prediction': str(prediction)})

