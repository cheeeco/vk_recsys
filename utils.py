import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lifelines.utils import concordance_index
from loguru import logger
from tqdm import tqdm

DUMPS_DIR = "dumps"
DATA_DIR = "data"
MODELS_DIR = "models"
SUBMISSIONS_DIR = "submissions"

VAL_USER_IDS_PATH = "dumps/val_user_ids.npy"
TRAIN_USER_IDS_PATH = "dumps/train_user_ids.npy"


def download_data_from_ods(data_dir: str = DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    urls = [
        "https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/VKRecsysChallenge2024/dataset/train_interactions.parquet",
        "https://storage.yandexcloud.net/ds-ods/files/files/c1992ccf/users_meta.parquet",
        "https://storage.yandexcloud.net/ds-ods/files/files/13b479ed/items_meta.parquet",
        "https://storage.yandexcloud.net/ds-ods/files/files/0235d298/test_pairs.csv",
        "https://storage.yandexcloud.net/ds-ods/files/files/55b07019/sample_submission.csv",
    ]
    logger.info("Started downloading data from ods")
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(data_dir, url.split("/")[-1])
            with open(filename, "wb") as file:
                file.write(response.content)
            logger.info(f"Downloaded: {filename}")
        else:
            logger.info(
                f"Failed to download: {url} (Status code: {response.status_code})"
            )


def load_data(data_dir: str = DATA_DIR):
    data_path = Path(data_dir)
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


def load_merged_data(data_dir: str = DATA_DIR):
    user_item_data, user_meta_data, item_meta_data, test_pairs_data = load_data(
        data_dir
    )
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
    user_item_data["timespent_rel"] = (
        user_item_data["timespent"] / user_item_data["duration"]
    )

    return user_item_data, test_pairs_data


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
            user_id=val_user_ids
            if len(target) == len(val_user_ids)
            else train_user_ids,
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
        probas = approxes / np.sum(approxes, axis=0)

        roc_auc_score = evaluate(
            user_id=val_user_ids
            if len(target) == len(val_user_ids)
            else train_user_ids,
            target=target,
            score=probas[2],
        )
        del train_user_ids, val_user_ids
        return roc_auc_score, 1

    def get_final_error(self, error, weight):
        return error / weight


def make_val_testlike(val_df: pd.DataFrame, target: str = "like") -> pd.DataFrame:
    val_df = val_df.copy()
    # filter users with less than 20 interactions
    interaction_counts = val_df["user_id"].value_counts()
    users_with_20_plus_interactions = interaction_counts[interaction_counts >= 20].index
    val_df = val_df[val_df["user_id"].isin(users_with_20_plus_interactions)]
    # keep only the first 20 interactions per user
    val_df["interaction_rank"] = val_df.groupby("user_id").cumcount() + 1
    val_df = val_df[val_df["interaction_rank"] <= 20].drop(columns="interaction_rank")
    # filter out users with zero total likes
    user_likes = val_df.groupby("user_id")[target].apply(lambda x: (x != 0).any())
    users_with_likes = user_likes[user_likes > 0].index
    val_df = val_df[val_df["user_id"].isin(users_with_likes)]

    return val_df
