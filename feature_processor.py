from pathlib import Path
from typing import Dict

import implicit
import lightfm
import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import csr_matrix


def feedback_group_features(
    user_item_data: pd.DataFrame,
    group_col: str = "user_id",
):
    """
    Computes aggregated feedback features grouped by a specified column.

    Parameters:
        user_item_data (pd.DataFrame): Input DataFrame containing user-item interaction data.
        group_col (str): Column by which to group the data (e.g., 'user_id', 'item_id').

    Returns:
        pd.DataFrame: DataFrame with aggregated feedback features, including counts and ratios for likes, dislikes, shares, and bookmarks.
    """
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


def get_viewers_statistics_by_item_id(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistics about the viewers of items, such as age and gender metrics.

    Parameters:
        history_df (pd.DataFrame): DataFrame containing user-item interaction data, including age and gender of viewers.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics on viewer age and gender for each item.
    """
    viewers_statistics_by_item_id = history_df.groupby("item_id").agg(
        min_viewers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="min"),
        max_viewers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="max"),
        mean_viewers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="mean"),
        median_viewers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="median"),
        std_viewers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="std"),
        min_viewers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="min"),
        max_viewers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="max"),
        mean_viewers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="mean"),
        median_viewers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="median"),
        std_viewers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="std"),
    )
    viewers_statistics_by_item_id = viewers_statistics_by_item_id.astype(
        {
            col_name: (np.float32 if dtype == np.float64 else dtype)
            for col_name, dtype in viewers_statistics_by_item_id.dtypes.to_dict().items()
        }
    )
    return viewers_statistics_by_item_id


def get_likers_statistics_by_item_id(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes statistics about the likers of items, such as age and gender metrics.

    Parameters:
        history_df (pd.DataFrame): DataFrame containing user-item interaction data, filtered for likers.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics on liker age and gender for each item.
    """
    likers_statistics_by_item_id = (
        history_df[history_df.like == 1]
        .groupby("item_id")
        .agg(
            min_likers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="min"),
            max_likers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="max"),
            mean_likers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="mean"),
            median_likers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="median"),
            std_likers_age_by_item_id=pd.NamedAgg(column="age", aggfunc="std"),
            min_likers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="min"),
            max_likers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="max"),
            mean_likers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="mean"),
            median_likers_gender_by_item_id=pd.NamedAgg(
                column="gender", aggfunc="median"
            ),
            std_likers_gender_by_item_id=pd.NamedAgg(column="gender", aggfunc="std"),
        )
    )
    likers_statistics_by_item_id = likers_statistics_by_item_id.astype(
        {
            col_name: (np.float32 if dtype == np.float64 else dtype)
            for col_name, dtype in likers_statistics_by_item_id.dtypes.to_dict().items()
        }
    )
    return likers_statistics_by_item_id


def compute_lag(
    train_df,
    val_df,
    group_col,
    target_col,
    lag,
):
    """
    Computes lag features for a specified target column grouped by a specified column.

    Parameters:
        train_df (pd.DataFrame): Training DataFrame where lag features are computed.
        val_df (pd.DataFrame): Validation DataFrame where lag features are merged.
        group_col (str): Column to group data by (e.g., 'user_id').
        target_col (str): Target column for which lag features are computed.
        lag (int): Number of periods to lag.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Updated training and validation DataFrames with lag features.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()
    grouped = train_df.groupby(group_col, as_index=False)
    lag_col_name = f"{target_col}_lag_{lag}"
    train_df[lag_col_name] = grouped[target_col].shift(lag).fillna(0).astype(np.int16)
    grouped = train_df.groupby(group_col, as_index=False)
    last_lag_by_group_col = grouped[lag_col_name].agg("last")
    val_df = val_df.merge(right=last_lag_by_group_col, on=group_col, how="left")
    return train_df, val_df


def merge_features_to_df(
    df: pd.DataFrame,
    user_group_features: pd.DataFrame,
    item_group_features: pd.DataFrame,
    source_group_features: pd.DataFrame,
    viewers_statistics_by_item_id: pd.DataFrame,
    likers_statistics_by_item_id: pd.DataFrame,
    user_view_counts_by_source_id_dict: Dict,
    user_like_counts_by_source_id_dict: Dict,
) -> pd.DataFrame:
    """
    Merges multiple feature DataFrames into the input DataFrame and computes additional derived features.

    Parameters:
        df (pd.DataFrame): Input DataFrame to merge features into.
        user_group_features (pd.DataFrame): Aggregated features by user.
        item_group_features (pd.DataFrame): Aggregated features by item.
        source_group_features (pd.DataFrame): Aggregated features by source.
        viewers_statistics_by_item_id (pd.DataFrame): Aggregated viewers statistics.
        likers_statistics_by_item_id (pd.DataFrame): Aggregated likerss statistics.
        user_view_counts_by_source_id_dict (Dict): Aggregated views counts by user_id and source_id.
        user_like_counts_by_source_id_dict (Dict): Aggregated likes counts by user_id and source_id.
    """
    df = df.merge(
        right=user_group_features,
        on="user_id",
        how="left",
    )
    df = df.merge(
        right=item_group_features,
        on="item_id",
        how="left",
    )
    df = df.merge(
        right=source_group_features,
        on="source_id",
        how="left",
    )
    df = df.merge(
        right=viewers_statistics_by_item_id,
        on="item_id",
        how="left",
    )
    df = df.merge(
        right=likers_statistics_by_item_id,
        on="item_id",
        how="left",
    )

    df["user_view_counts_by_source_id"] = [
        user_view_counts_by_source_id_dict.get((user_id, source_id), 0)
        for user_id, source_id in zip(df["user_id"], df["source_id"])
    ]
    df["user_like_counts_by_source_id"] = [
        user_like_counts_by_source_id_dict.get((user_id, source_id), 0)
        for user_id, source_id in zip(df["user_id"], df["source_id"])
    ]

    df["user_view_counts_by_source_id_ratio_to_views"] = (
        df["user_view_counts_by_source_id"] / df["num_of_views_by_user_id"]
    )
    df["user_like_counts_by_source_id_ratio_to_views"] = (
        df["user_like_counts_by_source_id"] / df["user_view_counts_by_source_id"]
    )
    df["user_like_counts_by_source_id_ratio_to_likes"] = (
        df["user_like_counts_by_source_id"] / df["num_of_likes_by_user_id"]
    )

    df["user_view_counts_by_source_id_ratio_to_views"] = df[
        "user_view_counts_by_source_id_ratio_to_views"
    ].fillna(0)
    df["user_like_counts_by_source_id_ratio_to_views"] = df[
        "user_like_counts_by_source_id_ratio_to_views"
    ].fillna(0)
    df["user_like_counts_by_source_id_ratio_to_likes"] = df[
        "user_like_counts_by_source_id_ratio_to_likes"
    ].fillna(0)
    df = df.astype(
        {
            "user_like_counts_by_source_id": np.int16,
            "user_view_counts_by_source_id": np.int16,
            "user_view_counts_by_source_id_ratio_to_views": np.float32,
            "user_like_counts_by_source_id_ratio_to_views": np.float32,
            "user_like_counts_by_source_id_ratio_to_likes": np.float32,
        }
    )
    return df


def make_collaborative_filtering(
    history_df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    check_for_dumps: bool = True,
    include_lightfm_scores: bool = True,
):
    """
    Train CF model(s) on history_df and add scores to train_df, val_df and test_df.

    Parameters:
        history_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for features computation and training 1st stage model.
        train_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for training 2nd stage models (catboost).
        val_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for validating 2nd stage models (catboost).
        test_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for making submission by 2nd stage models (catboost).
    """
    TRAIN_IALS_SCORE_PATH = "dumps/train_ials_score.npy"
    VAL_IALS_SCORE_PATH = "dumps/val_ials_score.npy"
    TEST_IALS_SCORE_PATH = "dumps/test_ials_score.npy"

    TRAIN_LFM_SCORE_PATH = "dumps/train_lfm_score.npy"
    VAL_LFM_SCORE_PATH = "dumps/val_lfm_score.npy"
    TEST_LFM_SCORE_PATH = "dumps/test_lfm_score.npy"

    history_df = history_df.copy()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    history_df["timespent_rel"] = history_df["timespent"] / history_df["duration"]

    share_weight = 10
    bookmarks_weight = 1
    timespent_rel_weight = 50
    history_df["weighted_like"] = history_df["like"] * (
        1
        + share_weight * history_df.share
        + bookmarks_weight * history_df.bookmarks
        + timespent_rel_weight * history_df.timespent_rel
    )

    if check_for_dumps and all(
        [
            Path(filepath).exists()
            for filepath in [
                TRAIN_IALS_SCORE_PATH,
                VAL_IALS_SCORE_PATH,
                TEST_IALS_SCORE_PATH,
            ]
        ]
    ):
        logger.info("Found dumps")
        train_ials_score = np.load(TRAIN_IALS_SCORE_PATH)
        val_ials_score = np.load(VAL_IALS_SCORE_PATH)
        test_ials_score = np.load(TEST_IALS_SCORE_PATH)
    else:
        u_history = history_df.user_id
        i_history = history_df.item_id
        weighted_like_istory = history_df.like

        sparse_history = csr_matrix((weighted_like_istory, (u_history, i_history)))
        sparse_history = implicit.nearest_neighbours.tfidf_weight(sparse_history)

        model = implicit.als.AlternatingLeastSquares(
            random_state=42,
        )
        model.fit(sparse_history, show_progress=True)
        model = model.to_cpu()

        train_ials_score = (
            model.user_factors[train_df["user_id"]]
            * model.item_factors[train_df["item_id"]]
        ).sum(axis=1)
        val_ials_score = (
            model.user_factors[val_df["user_id"]]
            * model.item_factors[val_df["item_id"]]
        ).sum(axis=1)
        test_ials_score = (
            model.user_factors[test_df["user_id"]]
            * model.item_factors[test_df["item_id"]]
        ).sum(axis=1)

        del model

        np.save(TRAIN_IALS_SCORE_PATH, train_ials_score)
        np.save(VAL_IALS_SCORE_PATH, val_ials_score)
        np.save(TEST_IALS_SCORE_PATH, test_ials_score)

    train_df["ials_score"] = train_ials_score[: len(train_df)]
    val_df["ials_score"] = val_ials_score[: len(val_df)]
    test_df["ials_score"] = test_ials_score

    if include_lightfm_scores:
        if check_for_dumps and all(
            [
                Path(filepath).exists()
                for filepath in [
                    TRAIN_LFM_SCORE_PATH,
                    VAL_LFM_SCORE_PATH,
                    TEST_LFM_SCORE_PATH,
                ]
            ]
        ):
            logger.info("Found dumps")
            train_lfm_score = np.load(TRAIN_LFM_SCORE_PATH)
            val_lfm_score = np.load(VAL_LFM_SCORE_PATH)
            test_lfm_score = np.load(TEST_LFM_SCORE_PATH)
        else:
            u_history = history_df.user_id
            i_history = history_df.item_id
            weighted_like_istory = history_df.like-history_df.dislike

            sparse_history = csr_matrix((weighted_like_istory, (u_history, i_history)))

            model = lightfm.LightFM(no_components=128, loss="bpr", random_state=42)
            model.fit(
                interactions=sparse_history, epochs=40, num_threads=16, verbose=True
            )

            train_lfm_score = model.predict(train_df["user_id"].values, train_df["item_id"].values, num_threads=16)
            val_lfm_score = model.predict(val_df["user_id"].values, val_df["item_id"].values, num_threads=16)
            test_lfm_score = model.predict(test_df["user_id"].values, test_df["item_id"].values, num_threads=16)

            del model

            np.save(TRAIN_LFM_SCORE_PATH, train_lfm_score)
            np.save(VAL_LFM_SCORE_PATH, val_lfm_score)
            np.save(TEST_LFM_SCORE_PATH, test_lfm_score)

        train_df["lfm_score"] = train_lfm_score[: len(train_df)]
        val_df["lfm_score"] = val_lfm_score[: len(val_df)]
        test_df["lfm_score"] = test_lfm_score

    return train_df, val_df, test_df


def create_features(
    history_df: pd.DataFrame,
    train_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    check_for_dumps: bool = True,
):
    """
    Calculates group and (in future) lag features, CF features based on history_df for train_df.

    Parameters:
        history_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for features computation and training 1st stage model.
        train_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for training 2nd stage models (catboost).
        val_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for validating 2nd stage models (catboost).
        test_df (pd.DataFrame): DataFrame with history of user-item interactions. Is used for making submission by 2nd stage models (catboost).
    """
    TRAIN_DF_PATH = "dumps/train_df.parquet"
    VAL_DF_PATH = "dumps/val_df.parquet"
    TEST_DF_PATH = "dumps/test_df.parquet"
    # check for dump
    if check_for_dumps and all(
        [
            Path(filepath).exists()
            for filepath in [TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH]
        ]
    ):
        train_df = pd.read_parquet(TRAIN_DF_PATH)
        val_df = pd.read_parquet(VAL_DF_PATH)
        test_df = pd.read_parquet(TEST_DF_PATH)
    else:
        # views and feedback grouped by user_id
        user_group_features = feedback_group_features(
            user_item_data=history_df, group_col="user_id"
        )
        """
        'num_of_views_by_user_id',
        'num_of_likes_by_user_id',
        'ratio_of_likes_by_user_id',
        'num_of_dislikes_by_user_id',
        'ratio_of_dislikes_by_user_id',
        'num_of_shares_by_user_id',
        'ratio_of_shares_by_user_id',
        'num_of_bookmarks_by_user_id',
        'ratio_of_bookmarks_by_user_id',
        """

        # views and feedback grouped by item_id
        item_group_features = feedback_group_features(
            user_item_data=history_df, group_col="item_id"
        )
        """
        'num_of_views_by_item_id',
        'num_of_likes_by_item_id',
        'ratio_of_likes_by_item_id',
        'num_of_dislikes_by_item_id',
        'ratio_of_dislikes_by_item_id',
        'num_of_shares_by_item_id',
        'ratio_of_shares_by_item_id',
        'num_of_bookmarks_by_item_id',
        'ratio_of_bookmarks_by_item_id',
        """

        # views and feedback grouped by source_id
        source_group_features = feedback_group_features(
            user_item_data=history_df, group_col="source_id"
        )
        """
        'num_of_views_by_source_id',
        'num_of_likes_by_source_id',
        'ratio_of_likes_by_source_id',
        'num_of_dislikes_by_source_id',
        'ratio_of_dislikes_by_source_id',
        'num_of_shares_by_source_id',
        'ratio_of_shares_by_source_id',
        'num_of_bookmarks_by_source_id',
        'ratio_of_bookmarks_by_source_id',
        """

        # item_id average gender and age of viewer
        viewers_statistics_by_item_id = get_viewers_statistics_by_item_id(
            history_df=history_df
        )
        """
        'min_viewers_age_by_item_id',
        'max_viewers_age_by_item_id',
        'mean_viewers_age_by_item_id',
        'median_viewers_age_by_item_id',
        'std_viewers_age_by_item_id',
        'min_viewers_gender_by_item_id',
        'max_viewers_gender_by_item_id',
        'mean_viewers_gender_by_item_id',
        'median_viewers_gender_by_item_id',
        'std_viewers_gender_by_item_id',
        """

        # item_id average gender and age of liker
        likers_statistics_by_item_id = get_likers_statistics_by_item_id(
            history_df=history_df
        )
        """
        'min_likers_age_by_item_id',
        'max_likers_age_by_item_id',
        'mean_likers_age_by_item_id',
        'median_likers_age_by_item_id',
        'std_likers_age_by_item_id',
        'min_likers_gender_by_item_id',
        'max_likers_gender_by_item_id',
        'mean_likers_gender_by_item_id',
        'median_likers_gender_by_item_id',
        'std_likers_gender_by_item_id',
        """

        # source_id-specific views grouped by user_id
        user_view_counts_by_source_id = history_df.groupby(
            by="user_id"
        ).source_id.apply(lambda x: x.value_counts())
        user_view_counts_by_source_id_dict = user_view_counts_by_source_id.to_dict()

        # source_id-specific feedback grouped by user_id
        user_like_counts_by_source_id = (
            history_df[history_df.like == 1]
            .groupby(by="user_id")
            .source_id.apply(lambda x: x.value_counts())
        )
        user_like_counts_by_source_id_dict = user_like_counts_by_source_id.to_dict()

        # MERGE
        if train_df is None:
            pass
        else:
            train_df = merge_features_to_df(
                df=train_df,
                user_group_features=user_group_features,
                item_group_features=item_group_features,
                source_group_features=source_group_features,
                viewers_statistics_by_item_id=viewers_statistics_by_item_id,
                likers_statistics_by_item_id=likers_statistics_by_item_id,
                user_view_counts_by_source_id_dict=user_view_counts_by_source_id_dict,
                user_like_counts_by_source_id_dict=user_like_counts_by_source_id_dict,
            )

        if val_df is None:
            pass
        else:
            val_df = merge_features_to_df(
                df=val_df,
                user_group_features=user_group_features,
                item_group_features=item_group_features,
                source_group_features=source_group_features,
                viewers_statistics_by_item_id=viewers_statistics_by_item_id,
                likers_statistics_by_item_id=likers_statistics_by_item_id,
                user_view_counts_by_source_id_dict=user_view_counts_by_source_id_dict,
                user_like_counts_by_source_id_dict=user_like_counts_by_source_id_dict,
            )

        if test_df is None:
            pass
        else:
            test_df = merge_features_to_df(
                df=test_df,
                user_group_features=user_group_features,
                item_group_features=item_group_features,
                source_group_features=source_group_features,
                viewers_statistics_by_item_id=viewers_statistics_by_item_id,
                likers_statistics_by_item_id=likers_statistics_by_item_id,
                user_view_counts_by_source_id_dict=user_view_counts_by_source_id_dict,
                user_like_counts_by_source_id_dict=user_like_counts_by_source_id_dict,
            )

        train_df.to_parquet(TRAIN_DF_PATH)
        val_df.to_parquet(VAL_DF_PATH)
        test_df.to_parquet(TEST_DF_PATH)

    return train_df, val_df, test_df
