import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from loguru import logger

from feature_processor import create_features, make_collaborative_filtering
from utils import (
    TRAIN_USER_IDS_PATH,
    VAL_USER_IDS_PATH,
    ROCAUCMetric,
    evaluate,
    download_data_from_ods,
    load_merged_data,
    make_val_testlike,
)

MODEL_TYPE = "ranker"
MODEL_CKPT_PATH = "models/ranker_008_uptrained"
SUBMISSION_COLUMNS = ["user_id", "item_id", "predict"]
SUBMISSION_PATH = "submissions/ranker_008_uptrained.csv"


if __name__ == "__main__":
    logger.info("Pipeline launched")

    download_data_from_ods()
    logger.info("Downloaded data from ods")

    user_item_data, test_pairs_data = load_merged_data()
    logger.info("Data loaded")

    target = "explicit"

    n_interactions = user_item_data.shape[0]
    val_size = int(n_interactions * 0.15)
    train_size = int(10e6)
    history_size = n_interactions - train_size - val_size

    history_df = user_item_data[:history_size]
    train_df = user_item_data[history_size : history_size + train_size]
    val_df = user_item_data[-val_size:]
    del user_item_data
    logger.info(f"Splitted data: {history_size=}, {train_size=}, {val_size=}")

    train_df, val_df, test_df = make_collaborative_filtering(
        history_df=history_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_pairs_data,
        check_for_dumps=True,
        include_lightfm_scores=True,
    )
    logger.info("Made collaborative filtering")

    train_df, val_df, test_df = create_features(
        history_df=history_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        check_for_dumps=True,
    )
    del history_df
    logger.info("Created features")

    columns = list(
        set(train_df.columns.to_list())
        - set(
            [
                "bookmarks",
                "dislike",
                "explicit",
                "like",
                "share",
                "timespent",
                "timespent_rel",
            ]
        )
    )

    ranker = CatBoostRanker(
        verbose=True,
        iterations=5000,
        loss_function="PairLogitPairwise",
        cat_features=["user_id", "source_id", "item_id"],
        task_type="GPU",
        devices="0",
        eval_metric=ROCAUCMetric(),
        metric_period=100,
    )
    logger.info(f"Initialized model: {ranker=}")

    train_df = train_df.sort_values(by="user_id", axis=0)
    val_df = make_val_testlike(val_df, "explicit")
    val_df = val_df.sort_values(by="user_id", axis=0)

    val_pool = Pool(
        data=val_df[columns],
        label=val_df[target],
        group_id=val_df["user_id"],
        cat_features=["user_id", "source_id", "item_id"],
    )

    np.save(TRAIN_USER_IDS_PATH, train_df.user_id.values)
    np.save(VAL_USER_IDS_PATH, val_df.user_id.values)

    ranker.fit(
        X=train_df[columns],
        y=train_df[target],
        group_id=train_df["user_id"],
        eval_set=val_pool,
    )
    logger.info("Successfully finished training")

    ranker.save_model(MODEL_CKPT_PATH)
    logger.info("Saved model")

    ranker_prediction = ranker.predict(val_df[columns])
    logger.info("Predicted val scores")

    ranker_score = evaluate(
        val_df.user_id.values, val_df[target].values, ranker_prediction
    )
    logger.info(f"Evaluated: {ranker_score=}")

    test_score = ranker.predict(test_df[columns])
    logger.info("Predicted test scores")

    if len(test_score.shape) > 1 and test_score.shape[-1] > 1:
        # it is classifier
        test_score = test_score[:, 1]
    test_df["predict"] = test_score
    test_df[SUBMISSION_COLUMNS].to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Saved submission at {SUBMISSION_PATH}")

    feature_names = ranker.feature_names_
    feature_importances = ranker.feature_importances_
    feat_imp_df = pd.DataFrame(
        {"name": feature_names, "importance": feature_importances}
    )
    feat_imp_df.to_csv("feature_importances.csv")
