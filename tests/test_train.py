import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_train_model_file_exists():
    """Vérifie que le fichier logistic_regression_model.pkl est créé après exécution de train.py"""
    assert os.path.exists('logistic_regression_model.pkl'), (
        "Le fichier logistic_regression_model.pkl n'existe pas après l'exécution de train.py."
    )

def test_train_model_loading():
    """Vérifie que le fichier sauvegardé contient uune regression logistique"""
    model = joblib.load('logistic_regression_model.pkl')
    assert isinstance(model, LogisticRegression), (
        "Le fichier logistic_regression_model.pkl ne contient pas un modèle LogisticRegression."
    )

def test_train_model_prediction():
    """Vérifie que le modèle entraîné peut prédire sur un sous-ensemble des données"""
    model = joblib.load('logistic_regression_model.pkl')
    data = pd.read_csv('data/customer_churn.csv')
    X = data[['Age','Total_Purchase','Account_Manager','Years','Num_Sites']]

    prediction = model.predict(X[:1])
    assert prediction is not None, "Le modèle n'a pas retourné de prédiction."

