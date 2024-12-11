import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker

MODEL_TYPE = "ranker"
MODEL_CKPT_PATH = "ranker"
TEST_DATASET_PARQUET_PATH = "data/test_pairs_data_w_group_features.parquet"
SUBMISSION_COLUMNS = ["user_id", "item_id", "predict"]
SUBMISSION_PATH = "submissions/cbr_group_fs_cat_submission.csv"

model_mapper = {
    "ranker": CatBoostRanker,
    "classifier": CatBoostClassifier,
}

if __name__ == "__main__":
    test_pairs_data = pd.read_parquet(TEST_DATASET_PARQUET_PATH)
    print("LOADED DATA")
    model = model_mapper[MODEL_TYPE]()
    model.load_model(MODEL_CKPT_PATH)
    print("LOADED MODEL")
    test_proba = model.predict(test_pairs_data[model.feature_names_])
    print("MADE PREDICTION")
    if len(test_proba.shape) > 1 and test_proba.shape[-1] > 1:
        # it is classifier
        test_proba = test_proba[:, 1]
    test_pairs_data["predict"] = test_proba
    test_pairs_data[SUBMISSION_COLUMNS].to_csv(SUBMISSION_PATH, index=False)
    print(f"SAVED SUBMISSION AT {SUBMISSION_PATH}")
