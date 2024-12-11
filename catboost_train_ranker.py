import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from utils import evaluate, ROCAUCMetric, VAL_USER_IDS_PATH

if __name__ == "__main__":
    user_item_data_w_group_features = pd.read_parquet(
        "data/user_item_data_w_group_features.parquet"
    )
    columns = list(
        set(user_item_data_w_group_features.columns.to_list())
        - set(
            [
                "bookmarks",
                "dislike",
                "explicit",
                "like",
                "share",
                "timespent",
            ]
        )
    )

    target = "explicit"
    shape = user_item_data_w_group_features.shape
    train_size = int(10e6)
    val_size = int(shape[0] * 0.15)
    user_item_data_w_group_features = user_item_data_w_group_features[
        -(train_size + val_size) :
    ]

    val_df = user_item_data_w_group_features[columns + [target]][
        -int(len(user_item_data_w_group_features) * 0.15) :
    ]
    train_df = user_item_data_w_group_features[columns + [target]][
        -int(len(user_item_data_w_group_features) * 0.15) - int(10e6) : -int(
            len(user_item_data_w_group_features) * 0.15
        )
    ]

    train_df = user_item_data_w_group_features[columns + [target]][-int(10e6) :]

    del user_item_data_w_group_features

    ranker = CatBoostRanker(
        verbose=True,
        iterations=1000,
        cat_features=["user_id", "source_id", "item_id"],
        # task_type="GPU",
        # devices="0",
        eval_metric=ROCAUCMetric(),
        metric_period=10,
    )
    train_df = train_df.sort_values(by="user_id", axis=0)
    val_df = val_df.sort_values(by="user_id", axis=0)

    val_pool = Pool(
        data=val_df[columns],
        label=val_df[target],
        group_id=val_df["user_id"],
        cat_features=["user_id", "source_id", "item_id"],
    )

    np.save(VAL_USER_IDS_PATH, val_df.user_id.values)
    
    ranker.fit(
        X=train_df[columns],
        y=train_df[target],
        group_id=train_df["user_id"],
        eval_set=val_pool
    )
    print("ended training")
    ranker.save_model("ranker")
    print("saved model")
    ranker_prediction = ranker.predict(val_df[:])
    print("predicted")
    ranker_score = evaluate(
        val_df.user_id.values, val_df[target].values, ranker_prediction
    )
    print(f"{ranker_score=}")
