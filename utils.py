import numpy as np
from tqdm import tqdm
from lifelines.utils import concordance_index


def evaluate(user_id: np.ndarray, target: np.ndarray, score: np.ndarray) -> np.float64:
    sorting_indices = user_id.argsort()

    user_id = user_id[sorting_indices]
    target_and_score = np.stack([target,score]).reshape(-1,2)
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