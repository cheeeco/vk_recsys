import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from loguru import logger

from feature_processor import create_features, make_collaborative_filtering
from utils import (
    TRAIN_USER_IDS_PATH,
    VAL_USER_IDS_PATH,
    ROCAUCMetric,
    evaluate,
    load_merged_data,
    make_val_testlike,
)

MODEL_TYPE = "classifier"
MODEL_CKPT_PATH = "models/classifier_007"
TEST_DATASET_PARQUET_PATH = "data/test_pairs_data_w_group_features.parquet"
SUBMISSION_COLUMNS = ["user_id", "item_id", "predict"]
SUBMISSION_PATH = "submissions/classifier_007_uptrained.csv"

model_mapper = {
    "ranker": CatBoostRanker,
    "classifier": CatBoostClassifier,
}

if __name__ == "__main__":
    # test_pairs_data = pd.read_parquet(TEST_DATASET_PARQUET_PATH)
    # print("LOADED DATA")
    logger.info("Pipeline launched")

    user_item_data, test_pairs_data = load_merged_data()
    logger.info("Data loaded")

    target = "explicit"

    n_interactions = user_item_data.shape[0]
    val_size = 1  # int(n_interactions * 0.15)
    train_size = 1  # int(10e6)
    history_size = n_interactions - train_size - val_size

    history_df = user_item_data[:history_size]
    train_df = user_item_data[history_size : history_size + train_size]
    val_df = user_item_data[-val_size:]
    del user_item_data
    logger.info(f"Splitted data: {history_size=}, {train_size=}, {val_size=}")

    train_df, val_df, test_df = create_features(
        history_df=history_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_pairs_data,
        check_for_dumps=False,
    )
    logger.info("Created features")

    train_df, val_df, test_df = make_collaborative_filtering(
        history_df=history_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        check_for_dumps=False,
    )
    del history_df, train_df, val_df
    logger.info("Made collaborative filtering")

    model = model_mapper[MODEL_TYPE]()
    model.load_model(MODEL_CKPT_PATH)
    print("LOADED MODEL")
    test_pairs_data = test_df
    test_proba = model.predict(test_pairs_data[model.feature_names_])
    print("MADE PREDICTION")
    if len(test_proba.shape) > 1 and test_proba.shape[-1] > 1:
        # it is classifier
        test_proba = test_proba[:, 1]
    test_pairs_data["predict"] = test_proba
    test_pairs_data[SUBMISSION_COLUMNS].to_csv(SUBMISSION_PATH, index=False)
    print(f"SAVED SUBMISSION AT {SUBMISSION_PATH}")
