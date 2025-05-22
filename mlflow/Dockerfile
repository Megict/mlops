FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD mlflow run . --entry-point data-preprocessing --env-manager local --experiment-name Wine_Project --run-name Data_Preprocessing -P test-size=0.3 && \
    mlflow run . --entry-point hyperparameters-tuning --env-manager local --experiment-name Wine_Project --run-name Hyperparameters_Search -P n-trials=10 && \
    mlflow run . --entry-point model-training --env-manager local --experiment-name Wine_Project --run-name Model_Training && \ 
    mlflow ui