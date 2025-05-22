import tempfile
import sys
import os
import argparse
import logging
import warnings
import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from xgboost.callback import TrainingCallback
from loguru import logger

## Логирование
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
warnings.filterwarnings('ignore')
logging.getLogger('mlflow').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.ERROR)

class LoggingCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        for metric_name, metric_vals in evals_log['test'].items():
            mlflow.log_metric(f"{metric_name}", metric_vals[-1][0], step=epoch)
        return False


# оптимизация с помощью OPTUNA
def objective(trial):
    global dtrain

    params = {
        "objective": trial.suggest_categorical('objective', ['multi:softmax']),
        "num_class": trial.suggest_categorical('num_class', [3]),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "alpha": trial.suggest_float("alpha", 0.001, 0.05),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        "num_boost_round": trial.suggest_int("num_boost_round", 30, 300),
    }

    with mlflow.start_run(nested=True):

        mlflow.log_params(params)
        params.update(eval_metric=['auc', 'merror'])
        num_boost_round = params["num_boost_round"]
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            nfold=3,
            callbacks=[LoggingCallback()],
            verbose_eval=False,
        )

        error = cv_results['test-merror-mean'].iloc[-1]
        mlflow.log_metric("accuracy", (1 - error))
        logger.info(f"Attempt: {trial.number}, Accuracy: {1 - error}")

        return error


if __name__ == '__main__':

    n_trials = 10
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", default=n_trials, type=float)
    n_trials = parser.parse_args().n_trials

    logger.info(f'Перебор гиперпараметров с {n_trials} попытками')

    with mlflow.start_run() as run:
        
        experiment_id = run.info.experiment_id


        last_run_id = mlflow.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName = 'Data_Preprocessing' and status = 'FINISHED'",
            order_by=["start_time DESC"]
        ).loc[0, 'run_id']

        with tempfile.TemporaryDirectory() as tmpdir:
            mlflow.artifacts.download_artifacts(run_id=last_run_id, artifact_path='datasets/train.csv', dst_path=tmpdir)
            train = pd.read_csv(os.path.join(tmpdir, 'datasets/train.csv'))


        features = [i for i in train.columns if i != 'target']
        dtrain = xgb.DMatrix(data=train.loc[:, features], label=train['target'])

        # logger.info(train)
        logger.info('Starting optuna study')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial

        logger.info(f'Optimization finished, best params: {best_trial.params}')
        mlflow.log_params(best_trial.params)

        logger.info(f'Best trial Accuracy: {1 - best_trial.value}')
        mlflow.log_metric('accuracy', 1 - study.best_value)