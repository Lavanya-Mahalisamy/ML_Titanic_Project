import joblib
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.linear_model import LogisticRegression

df=sns.load_dataset("titanic")
df=df[["pclass","sex","age","fare","survived"]]
x=df.drop("survived",axis=1)
y=df["survived"]
numerical_features=["age","fare"]
categorical_features=["sex","pclass"]
numerical_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
categorical_pipeline=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OrdinalEncoder())])
preprocessor=ColumnTransformer([("num",numerical_pipeline,numerical_features),("cat",categorical_pipeline,categorical_features)])
pipeline=Pipeline([("preprocessor",preprocessor),("classifier",LogisticRegression())])
pipeline.fit(x,y)
os.makedirs("model",exist_ok=True)
joblib.dump(pipeline,"model/pipeline.pkl")
print("Pipeline saved successfully.")
