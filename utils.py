from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines.utils import concordance_index
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

VAL_USER_IDS_PATH = "dumps/val_user_ids.npy"
TRAIN_USER_IDS_PATH = "dumps/train_user_ids.npy"


def load_data():
    data_path = Path("./data")
    user_item_data_path = data_path / "train_interactions.parquet"
    users_meta_data_path = data_path / "users_meta.parquet"
    items_meta_data_path = data_path / "items_meta.parquet"
    test_pairs_data_path = data_path / "test_pairs.csv"

    user_item_data = pd.read_parquet(user_item_data_path)
    user_meta_data = pd.read_parquet(users_meta_data_path)
    item_meta_data = pd.read_parquet(items_meta_data_path)
    test_pairs_data = pd.read_csv(test_pairs_data_path)

    # np.uint8 -> np.int16 cast to allow subtraction and just for predictable behaviour
    user_item_data[user_item_data.dtypes[user_item_data.dtypes == np.uint8].index] = (
        user_item_data[
            user_item_data.dtypes[user_item_data.dtypes == np.uint8].index
        ].astype(np.int16)
    )
    user_meta_data[user_meta_data.dtypes[user_meta_data.dtypes == np.uint8].index] = (
        user_meta_data[
            user_meta_data.dtypes[user_meta_data.dtypes == np.uint8].index
        ].astype(np.int16)
    )
    item_meta_data[item_meta_data.dtypes[item_meta_data.dtypes == np.uint8].index] = (
        item_meta_data[
            item_meta_data.dtypes[item_meta_data.dtypes == np.uint8].index
        ].astype(np.int16)
    )
    # single column for likes and dislikes
    user_item_data["explicit"] = user_item_data.like - user_item_data.dislike

    return user_item_data, user_meta_data, item_meta_data, test_pairs_data


def load_merged_data():
    user_item_data, user_meta_data, item_meta_data, test_pairs_data = load_data()
    user_item_data = user_item_data.merge(
        right=item_meta_data.drop(columns="embeddings"),
        on="item_id",
        how="left",
    )
    test_pairs_data = test_pairs_data.merge(
        right=item_meta_data.drop(columns="embeddings"),
        on="item_id",
        how="left",
    )
    user_item_data = user_item_data.merge(
        right=user_meta_data,
        on="user_id",
        how="left",
    )
    test_pairs_data = test_pairs_data.merge(
        right=user_meta_data,
        on="user_id",
        how="left",
    )
    user_item_data["timespent_rel"] = user_item_data["timespent"] / user_item_data["duration"]

    return user_item_data, test_pairs_data


def get_sparse_train_val(share_weight=0, bookmarks_weight=0, timespent_rel_weight=0):
    user_item_data, user_meta_data, item_meta_data, test_pairs_data = load_data()
    user_item_data = user_item_data.merge(
        item_meta_data.drop(columns="embeddings"), on="item_id", how="left"
    )
    user_item_data["timespent_rel"] = (
        user_item_data["timespent"] / user_item_data["duration"]
    )

    ui_train, ui_val = train_test_split(
        user_item_data, test_size=0.15, random_state=42, shuffle=False
    )

    ui_train["weighted_target"] = ui_train["like"] * (
        1
        + share_weight * ui_train.share
        + bookmarks_weight * ui_train.bookmarks
        + timespent_rel_weight * ui_train.timespent_rel
    )
    u_train = ui_train.user_id.values
    i_train = ui_train.item_id.values
    likes_train = ui_train.like.values
    # dislikes_train = ui_train.dislike

    u_val = ui_val.user_id.values
    i_val = ui_val.item_id.values
    likes_val = ui_val.like.values
    # dislikes_val = ui_val.dislike

    sparse_train = csr_matrix((likes_train, (u_train, i_train)))
    sparse_val = csr_matrix((likes_val, (u_val, i_val)))

    del (
        user_item_data,
        user_meta_data,
        item_meta_data,
        test_pairs_data,
        u_train,
        i_train,
        likes_train,
        # u_val,
        # i_val,
        # likes_val,
    )
    return sparse_train, sparse_val, u_val, i_val, likes_val


def evaluate(user_id: np.ndarray, target: np.ndarray, score: np.ndarray) -> np.float64:
    sorting_indices = user_id.argsort()

    user_id = user_id[sorting_indices]
    target_and_score = np.stack([target, score]).swapaxes(0, 1)
    target_and_score = target_and_score[sorting_indices]

    groups = np.split(target_and_score, np.unique(user_id, return_index=True)[1][1:])
    roc_aucs = []
    for group in tqdm(groups):
        target = group[:, 0]
        score = group[:, 1]
        if len(np.unique(target)) == 1:
            continue
        roc_auc = concordance_index(target, score)
        roc_aucs.append(roc_auc)
    return np.mean(roc_aucs)


def compare_score(score1, score2):
    if score1 > score2:
        return 0
    elif score1 == score2:
        return 0.5
    else:
        return 1


def compare_target(target1, target2):
    if target1 >= target2:
        return 0
    else:
        return 1


def roc_auc_score(y_true, y_score):
    num = 0
    denom = 0
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i == j:
                continue
            num += compare_score(y_score[i], y_score[j]) * compare_target(
                y_true[i], y_true[j]
            )
            denom += compare_target(y_true[i], y_true[j])
    return num / denom if denom > 0 else None


def plot_feature_importances(model, graphic=True, figsize=(10, 6), palette="viridis"):
    """
    Plots the feature importances of a trained CatBoost model.

    Parameters:
        model (catboost.CatBoost): Trained CatBoost model.
        graphic (bool): Flag for graphical visualization. If false, then feature importances are printed.
        figsize (tuple): Size of the plot, default is (10, 6).
        palette (str): Seaborn color palette for the barplot, default is 'viridis'.
    """
    # Get feature importances
    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_
    # Create a DataFrame for plotting
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    )

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(
        by="importance", ascending=False
    )

    if graphic is True:
        # Set up the matplotlib figure
        plt.figure(figsize=figsize)

        # Create a horizontal bar plot
        sns.barplot(
            x="importance",
            y="feature",
            data=feature_importance_df,
            palette=palette,
            hue="feature",
        )

        # Set plot title and labels
        plt.grid()
        plt.title("CatBoost Model Feature Importances", fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)

        # Display the plot
        plt.show()

    elif graphic is False:
        for feature, importance in zip(
            feature_importance_df["feature"], feature_importance_df["importance"]
        ):
            print(f"{feature=}, {importance=}")
    else:
        raise ValueError("ERROR. Wrong graphic argument")


def train_test_split_wise(
    user_item_data: pd.DataFrame,
    test_size: float = 0.15,
    random_state: int = 42,
    shuffle: bool = False,
):
    train_df, val_df = train_test_split(
        user_item_data,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    # Step 1: Filter users with less than 20 interactions
    interaction_counts = val_df["user_id"].value_counts()
    users_with_20_plus_interactions = interaction_counts[interaction_counts >= 20].index
    val_df = val_df[val_df["user_id"].isin(users_with_20_plus_interactions)]
    # Step 2: Keep only the first 20 interactions per user
    val_df["interaction_rank"] = val_df.groupby("user_id").cumcount() + 1
    val_df = val_df[val_df["interaction_rank"] <= 20].drop(columns="interaction_rank")
    # Step 3: Filter out users with zero total likes
    user_likes = val_df.groupby("user_id")["like"].sum()
    users_with_likes = user_likes[user_likes > 0].index
    val_df = val_df[val_df["user_id"].isin(users_with_likes)]

    return train_df, val_df


class ROCAUCMetric:
    def is_max_optimal(self):
        return True  # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        train_user_ids = np.load(TRAIN_USER_IDS_PATH)
        val_user_ids = np.load(VAL_USER_IDS_PATH)
        approx = approxes[0]

        roc_auc_score = evaluate(
            user_id=val_user_ids if len(target)==len(val_user_ids) else train_user_ids,
            target=target,
            score=approx,
        )
        del train_user_ids, val_user_ids
        return roc_auc_score, 1

    def get_final_error(self, error, weight):
        return error / weight


class ROCAUCMetric4Clf:
    def is_max_optimal(self):
        return True  # greater is better

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 3
        assert len(target) == len(approxes[0])

        train_user_ids = np.load(TRAIN_USER_IDS_PATH)
        val_user_ids = np.load(VAL_USER_IDS_PATH)

        approxes = np.exp(np.stack(approxes))
        probas = approxes/np.sum(approxes, axis=0)

        roc_auc_score = evaluate(
            user_id=val_user_ids if len(target)==len(val_user_ids) else train_user_ids,
            target=target,
            score=probas[2],
        )
        del train_user_ids, val_user_ids
        return roc_auc_score, 1

    def get_final_error(self, error, weight):
        return error / weight


def make_val_testlike(val_df: pd.DataFrame, target: str = "like") -> pd.DataFrame:
    val_df = val_df.copy()
    # Step 1: Filter users with less than 20 interactions
    interaction_counts = val_df["user_id"].value_counts()
    users_with_20_plus_interactions = interaction_counts[interaction_counts >= 20].index
    val_df = val_df[val_df["user_id"].isin(users_with_20_plus_interactions)]
    # Step 2: Keep only the first 20 interactions per user
    val_df["interaction_rank"] = val_df.groupby("user_id").cumcount() + 1
    val_df = val_df[val_df["interaction_rank"] <= 20].drop(columns="interaction_rank")
    # Step 3: Filter out users with zero total likes
    user_likes = val_df.groupby("user_id")[target].apply(lambda x: (x != 0).any())
    users_with_likes = user_likes[user_likes > 0].index
    val_df = val_df[val_df["user_id"].isin(users_with_likes)]

    return val_df
