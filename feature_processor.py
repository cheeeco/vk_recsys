import pandas as pd
import numpy as np


def like_dislike_group_features(
    user_item_data: pd.DataFrame,
    group_col: str = "user_id",
):
    # group_features = user_item_data.groupby(by=group_col).agg(
    #     num_of_views_by_user=pd.NamedAgg(column="like", aggfunc=len),
    #     num_of_likes_by_user=pd.NamedAgg(column="like", aggfunc="sum"),
    #     ratio_of_likes_by_user=pd.NamedAgg(column="like", aggfunc="mean"),
    #     num_of_dislikes_by_user=pd.NamedAgg(column="dislike", aggfunc="sum"),
    #     ratio_of_dislikes_by_user=pd.NamedAgg(column="dislike", aggfunc="mean"),
    # )
    # group_features = group_features.astype(
    #     {
    #         "num_of_views_by_user": np.int16,
    #         "num_of_likes_by_user": np.int16,
    #         "ratio_of_likes_by_user": np.float32,
    #         "num_of_dislikes_by_user": np.int16,
    #         "ratio_of_dislikes_by_user": np.float32,
    #     }
    # )
    # return group_features
    num_of_views_col = f"num_of_views_by_{group_col}"
    num_of_likes_col = f"num_of_likes_by_{group_col}"
    ratio_of_likes_col = f"ratio_of_likes_by_{group_col}"
    num_of_dislikes_col = f"num_of_dislikes_by_{group_col}"
    ratio_of_dislikes_col = f"ratio_of_dislikes_by_{group_col}"

    # Perform the aggregation
    group_features = user_item_data.groupby(by=group_col).agg(
        **{
            num_of_views_col: pd.NamedAgg(column="like", aggfunc=len),
            num_of_likes_col: pd.NamedAgg(column="like", aggfunc="sum"),
            ratio_of_likes_col: pd.NamedAgg(column="like", aggfunc="mean"),
            num_of_dislikes_col: pd.NamedAgg(column="dislike", aggfunc="sum"),
            ratio_of_dislikes_col: pd.NamedAgg(column="dislike", aggfunc="mean"),
        }
    )

    # Set data types
    group_features = group_features.astype(
        {
            num_of_views_col: np.int16,
            num_of_likes_col: np.int16,
            ratio_of_likes_col: np.float32,
            num_of_dislikes_col: np.int16,
            ratio_of_dislikes_col: np.float32,
        }
    )

    return group_features
