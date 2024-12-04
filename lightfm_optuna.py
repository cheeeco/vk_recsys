import numpy as np
import optuna
import lightfm
from scipy.sparse import csr_matrix
from loguru import logger
from optuna.storages import RDBStorage

from utils import evaluate, get_sparse_train_val


# Objective function for Optuna
def objective(trial):
    # Hyperparameter sampling
    no_components = trial.suggest_int("no_components", 8, 256, step=1)
    learning_schedule = trial.suggest_categorical("learning_schedule", ['adagrad', 'adadelta'])
    loss = trial.suggest_categorical("loss", ['logistic', 'bpr', 'warp', 'warp-kos'])
    regularization = trial.suggest_float("regularization", 1e-5, 9e-1, log=True)
    alpha = trial.suggest_loguniform("alpha", 1e-1, 10)
    iterations = trial.suggest_int("iterations", 15, 80, step=1)
    use_cg = trial.suggest_categorical("use_cg", [True, False])

    feedback_preprocessing = trial.suggest_categorical(
        "feedback_preprocessing", ["bm25_weight", "tfidf_weight"]
    )

    share_weight = trial.suggest_float("share_weight", 0.1, 500, log=True)
    bookmarks_weight = trial.suggest_float("bookmarks_weight", 0.1, 500, log=True)
    timespent_rel_weight = trial.suggest_float(
        "timespent_rel_weight", 0.1, 500, log=True
    )

    logger.info("Hyperparameters sampled")

    # Load data
    sparse_train, sparse_val, u_val, i_val, likes_val = get_sparse_train_val(
        share_weight, bookmarks_weight, timespent_rel_weight
    )
    logger.info("Data loaded")

    # Preprocess data
    feedback_preprocessing = {
        "bm25_weight": bm25_weight,
        "tfidf_weight": tfidf_weight,
        "None": None,
    }[feedback_preprocessing]

    sparse_train = (
        feedback_preprocessing(sparse_train)
        if feedback_preprocessing is not None
        else sparse_train
    )
    logger.info(f"Data preprocessed with {feedback_preprocessing=}")

    # Create ALS model with sampled parameters
    model = lightfm.LightFM(
        no_components=128,
        loss="bpr",
        random_state=42,
    )
    logger.info("Model initialized")

    # Train the model
    model.fit(sparse_train, show_progress=True)
    logger.info("Model trained")

    # Predict scores
    model = model.to_cpu()
    als_pred = (model.user_factors[u_val] * model.item_factors[i_val]).sum(axis=1)
    logger.info("Scores predicted")

    # Evaluate the model on the train data (for simplicity here; use a proper train-test split)
    roc_auc_score = evaluate(user_id=u_val, target=likes_val, score=als_pred)
    logger.info("ROC AUC calculated")

    return roc_auc_score


if __name__ == "__main__":
    logger.info("WELCOME")

    # Use SQLite storage to persist the study
    storage = RDBStorage("sqlite:///optuna_study.db")
    logger.info("SQLite storage initialized")

    # Run Optuna optimization
    optuna.delete_study(study_name="als_weighted_optimization", storage=storage)
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="als_weighted_optimization",
        load_if_exists=True,
    )
    logger.info("Study initialized")

    study.optimize(objective, n_trials=50)
    logger.info("Study optimized")

    # Dump study to csv
    df = study.trials_dataframe()
    df.to_csv("optuna_als_trials.csv", index=False)
    logger.info("Trials saved to optuna_trials.csv")

    # Best hyperparameters
    logger.info(f"{study.best_params=}, {study.best_value=}")
