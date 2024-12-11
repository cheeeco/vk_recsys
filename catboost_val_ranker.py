import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from utils import evaluate, make_val_testlike

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

    del user_item_data_w_group_features

    ranker = CatBoostRanker(
        verbose=True,
    )

    val_df = make_val_testlike(val_df=val_df, target="explicit")
    val_df = val_df.sort_values(by="user_id", axis=0)

    ranker.load_model("ranker")
    print("loaded model")
    ranker_prediction = ranker.predict(val_df[ranker.feature_names_])
    print("predicted")
    ranker_score = evaluate(
        val_df.user_id.values, val_df[target].values, ranker_prediction
    )
    print(f"{ranker_score=}")
