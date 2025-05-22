## Проект mlflow.

Этот проект демонстрирует полный цикл машинного обучения для классификации вина с использованием:
- MLflow для отслеживания экспериментов и управления моделями
- XGBoost для построения моделей
- Optuna для оптимизации гиперпараметров

Проект включает три основных этапа:
1. Предобработку данных
2. Подбор гиперпараметров
3. Обучение и регистрацию модели

Запуск проекта через MLflow:
```bash
mlflow run . --entry-point data-preprocessing
mlflow run . --entry-point hyperparameters-tuning
mlflow run . --entry-point model-training
```
Запуск через Docker

```bash
docker build -t wine-classification .
docker run -p 5000:5000 wine-classification
```

После запуска интерфейс MLflow доступен по адресу: `http://localhost:5000`

### Скриншоты MLflow ui.

![изображение](https://github.com/user-attachments/assets/63fb5897-fbce-406b-ab65-bca5724cee57)

![изображение](https://github.com/user-attachments/assets/f5404dd7-e8e7-4e3c-9345-4bc7cdfaa0a5)

![изображение](https://github.com/user-attachments/assets/b1cd0560-90f5-4f0d-89fb-04a59a9dc031)

![изображение](https://github.com/user-attachments/assets/4c84139b-7145-4d00-9f1c-48d145aa317d)

