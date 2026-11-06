import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "model-flan-t5-small-finetuned")
