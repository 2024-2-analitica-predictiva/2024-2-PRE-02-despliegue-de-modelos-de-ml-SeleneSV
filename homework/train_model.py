# train_model.py

"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

df = pd.read_csv("files/input/house_data.csv", sep=",") # Leer archivo csv

# Se separan las variables independientes del modelo
features = df[
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

# Columna target
target = df[["price"]]

# Crear modelo y entrenar
estimator = LinearRegression()
estimator.fit(features, target)

# Guardar modelo entrenado como objeto
with open("homework/house_predictor.pkl", "wb") as file:
    pickle.dump(estimator, file)