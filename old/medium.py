#%%
from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource
from sklearn.externals import joblib
import pickle
import numpy as np
import pandas as pd

#%%
app = Flask(__name__)
api = Api(app)

model_path = 'model.pkl'
with open(model_path, 'rb') as f:
    model = joblib.load(f)

model_columns_path = 'model_columns.pkl'
with open(model_columns_path, 'rb') as f:
    model_columns = joblib.load(f)

#%% Data process
def preprocess(df):
    # process into proper format
    df_ = pd.get_dummies(df)
    return df_.reindex(columns = model_columns, fill_value = 0)


#%% flask-restful argument parsing
parser = reqparse.RequestParser()
parser.add_argument('Age', action = 'append' )
parser.add_argument('Sex', action = 'append')
parser.add_argument('Embarked', action = 'append')

#%% flask restful app object
class PredictSentiment(Resource):
    def get(self):
        # parse args
        args = parser.parse_args()
        print(args) 

        # read into dataframe
        df = pd.DataFrame.from_dict(args)

        # process into proper format
        query = preprocess(df)

        # make prediction using preprocessed data
        prediction = list(model.predict(query))

        # create JSON object and return
        return jsonify({'prediction': str(prediction)})


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')


if __name__ == '__main__':
    app.run(debug=True)
