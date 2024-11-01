import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from lifelines.utils import concordance_index


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

    return user_item_data, user_meta_data, item_meta_data, test_pairs_data


def evaluate(user_id: np.ndarray, target: np.ndarray, score: np.ndarray) -> np.float64:
    sorting_indices = user_id.argsort()

    user_id = user_id[sorting_indices]
    target_and_score = np.stack([target,score]).swapaxes(0,1)
    target_and_score = target_and_score[sorting_indices]
    
    groups = np.split(target_and_score, np.unique(user_id, return_index=True)[1][1:])
    roc_aucs = []
    for group in tqdm(groups):
        target = group[:,0]
        score = group[:,1]
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
            num += compare_score(y_score[i], y_score[j])*compare_target(y_true[i], y_true[j])
            denom += compare_target(y_true[i], y_true[j])
    return num/denom if denom>0 else None