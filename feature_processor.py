import pandas as pd
import numpy as np


def feedback_group_features(
    user_item_data: pd.DataFrame,
    group_col: str = "user_id",
):
    num_of_views_col = f"num_of_views_by_{group_col}"
    num_of_likes_col = f"num_of_likes_by_{group_col}"
    ratio_of_likes_col = f"ratio_of_likes_by_{group_col}"
    num_of_dislikes_col = f"num_of_dislikes_by_{group_col}"
    ratio_of_dislikes_col = f"ratio_of_dislikes_by_{group_col}"
    num_of_shares_col = f"num_of_shares_by_{group_col}"
    ratio_of_shares_col = f"ratio_of_shares_by_{group_col}"
    num_of_bookmarks_col = f"num_of_bookmarks_by_{group_col}"
    ratio_of_bookmarks_col = f"ratio_of_bookmarks_by_{group_col}"

    # Perform the aggregation
    group_features = user_item_data.groupby(by=group_col).agg(
        **{
            num_of_views_col: pd.NamedAgg(column="like", aggfunc=len),
            num_of_likes_col: pd.NamedAgg(column="like", aggfunc="sum"),
            ratio_of_likes_col: pd.NamedAgg(column="like", aggfunc="mean"),
            num_of_dislikes_col: pd.NamedAgg(column="dislike", aggfunc="sum"),
            ratio_of_dislikes_col: pd.NamedAgg(column="dislike", aggfunc="mean"),
            num_of_shares_col: pd.NamedAgg(column="share", aggfunc="sum"),
            ratio_of_shares_col: pd.NamedAgg(column="share", aggfunc="mean"),
            num_of_bookmarks_col: pd.NamedAgg(column="bookmarks", aggfunc="sum"),
            ratio_of_bookmarks_col: pd.NamedAgg(column="bookmarks", aggfunc="mean"),
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
            num_of_shares_col: np.int16,
            ratio_of_shares_col: np.float32,
            num_of_bookmarks_col: np.int16,
            ratio_of_bookmarks_col: np.float32,
        }
    )

    return group_features


def compute_lag(
        train_df,
        val_df,
        group_col,
        target_col,
        lag,
        ):
    train_df = train_df.copy()
    val_df = val_df.copy()
    grouped = train_df.groupby(group_col, as_index=False)
    lag_col_name = f"{target_col}_lag_{lag}"
    train_df[lag_col_name] = grouped[target_col].shift(lag).fillna(0).astype(np.int16)
    grouped = train_df.groupby(group_col, as_index=False)
    last_lag_by_group_col = grouped[lag_col_name].agg("last")
    val_df = val_df.merge(
        right=last_lag_by_group_col,
        on=group_col,
        how="left"
    )
    return train_df, val_df