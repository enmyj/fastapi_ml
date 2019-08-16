#%% imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

#%% read in file and simplify 
fp = '/Users/ian.myjer/repos/ga_ian/2_dataset/titanic.csv'
df = pd.read_csv(fp)

df_ = df[['Age', 'Sex', 'Embarked', 'Survived']]


#%% preprocess (dummify cateogorical variables)
def preprocess(df):
     categoricals = []
     for col, col_type in df.dtypes.iteritems():
          if col_type == 'O':
               categoricals.append(col)
          else:
               df[col].fillna(0, inplace=True)

     return pd.get_dummies(df, columns=categoricals, dummy_na=True)

df_ohe = preprocess(df_)

#%% Fit model

target_var = ['Survived']
X = df_ohe[df_ohe.columns.difference(target_var)]
y = df_ohe[target_var]
lr = LogisticRegression().fit(X,y)

#%% Serialize

model_columns = list(X.columns)
dump(lr, '/Users/ian.myjer/repos/fastapi_ml/model.pkl')
dump(model_columns, '/Users/ian.myjer/repos/fastapi_ml/model_columns.pkl')
