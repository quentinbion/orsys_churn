# Utiliser une image Python légère
FROM python:3.8-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier des dépendances
COPY requirements.txt requirements.txt

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires dans l'image Docker
COPY train_log_reg.py train_log_reg.py
COPY test.py test.py
COPY flask_app.py flask_app.py
COPY data data
COPY tests tests

# Exposer le port utilisé par Flask
EXPOSE 5012

# Exécuter le script pour entraîner le modèle et générer churn_model_clean.pkl
RUN python train_log_reg.py

ENV PYTHONPATH=/app

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]

