import os
import sys
import tempfile
import mlflow
import warnings
import logging
import xgboost as xgb
import pandas as pd
from loguru import logger


# логирование
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


if __name__ == '__main__':

    logger.info('Model training started')
 
    mlflow.xgboost.autolog()

    with mlflow.start_run() as run:

        experiment_id = run.info.experiment_id
        run_id = run.info.run_id
        logger.info(f'Start mlflow run: {run_id}')
        
        # последний запуск препроцессинга
        last_data_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, 'run_id']
    
        # подгружаем датасеты
        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/train.csv', dst_path=tmpdir)
            mlflow.artifacts.download_artifacts(run_id=last_data_run_id, artifact_path='datasets/test.csv', dst_path=tmpdir)
            train = pd.read_csv(os.path.join(tmpdir, 'datasets/train.csv'))
            test = pd.read_csv(os.path.join(tmpdir, 'datasets/test.csv'))

        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])
        dtest = xgb.DMatrix(data=test.loc[:, features], label=test['target'])

        # последний запуск перебора
        last_tuning_run = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Hyperparameters_Search' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, :]
        
        params = {col.split('.')[1]: last_tuning_run[col] for col in last_tuning_run.index if 'params' in col}
        params.update(eval_metric=['auc', 'merror'])

        mlflow.log_params(params)
        
        model = xgb.train(
            dtrain=dtrain,
            num_boost_round=int(params["num_boost_round"]),
            params=params,
            evals=[(dtest, 'test')],
            verbose_eval=False,
            early_stopping_rounds=10
        )

        mlflow.log_metric("accuracy", 1 - model.best_score)
        
        input_example = test.loc[0:10, features]
        predictions_example = pd.DataFrame(model.predict(xgb.DMatrix(input_example)), columns=['predictions'])
        mlflow.xgboost.log_model(model, "booster", input_example=input_example)
        mlflow.log_text(predictions_example.to_json(orient='split', index=False), 'booster/predictions_example.json')


        model_uri = f"runs:/{run.info.run_id}/booster"
        mlflow.register_model(model_uri, 'WineModelBooster')
        
        # Log model as sklearn completable XGBClassifier
        params.update(num_boost_round=model.best_iteration)
        model = xgb.XGBClassifier(**params)
        model.fit(train.loc[:, features], train['target'])
        mlflow.xgboost.log_model(model, "model", input_example=input_example)


        mlflow.log_text(train.to_csv(index=False), 'datasets/train.csv')
        mlflow.log_text(test.to_csv(index=False),'datasets/test.csv')

        logger.info('Model training finished')


        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, 'CancerModel')
        
        logger.info('Model registered')