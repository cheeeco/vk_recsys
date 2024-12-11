import pandas as pd
from catboost import CatBoostClassifier

from utils import evaluate, ROCAUCMetric

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

    target = "like"
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

    del user_item_data_w_group_features

    model = CatBoostClassifier(
        verbose=True,
        iterations=100,
        auto_class_weights="SqrtBalanced",
        cat_features=["user_id", "source_id", "item_id"],
        task_type="GPU",
        devices="0",
        eval_metric=ROCAUCMetric(),
    )

    train_df = train_df.sort_values(by="user_id", axis=0)
    val_df = val_df.sort_values(by="user_id", axis=0)
    model.fit(
        X=train_df[columns],
        y=train_df[target],
        eval_set=(val_df[columns], val_df[target]),
    )
    print("ended training")
    model.save_model("classifier")
    print("saved model")
    classifier_prediction = model.predict_proba(val_df[columns])
    print("predicted")
    classifier_score = evaluate(
        val_df.user_id.values, val_df[target].values, classifier_prediction[:, 1]
    )
    print(f"{classifier_score=}")
