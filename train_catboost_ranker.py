import gc
import numpy as np
import pandas as pd
from copy import deepcopy
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import train_test_split

from utils import load_data, evaluate as my_evaluate

class ROCAUCMetric(object):
    def is_max_optimal(self):
        """
        Returns whether great values of metric are better
        """
        return True

    def evaluate(self, approxes, target, weight):
        """
        Evaluates metric value.

        Parameters
        ----------
        approxes : list of indexed containers (containers with only __len__ and __getitem__ defined) of float
            Vectors of approx labels.

        targets : one dimensional indexed container of float
            Vectors of true labels.

        weights : one dimensional indexed container of float, optional (default=None)
            Weight for each instance.

        Returns
        -------
            weighted error : float
            total weight : float

        """
        pass
    
    def get_final_error(self, error, weight):
        """
        Returns final value of metric based on error and weight.

        Parameters
        ----------
        error : float
            Sum of errors in all instances.

        weight : float
            Sum of weights of all instances.

        Returns
        -------
        metric value : float

        """
        pass


def fit_model(loss_function, additional_params=None, train_pool=None, test_pool=None):
    parameters = deepcopy(default_parameters)
    parameters["loss_function"] = loss_function
    parameters["train_dir"] = loss_function

    if additional_params is not None:
        parameters.update(additional_params)

    model = CatBoostRanker(**parameters)
    model.fit(train_pool, eval_set=test_pool, plot=True)

    return model


if __name__ == "__main__":
    user_item_data, user_meta_data, item_meta_data, test_pairs_data = load_data()

    user_embeddings = np.load("user_embeddings.npy")
    item_embeddings = np.load("item_embeddings.npy")
    user_biases = np.load("user_biases.npy")
    item_biases = np.load("item_biases.npy")
    lightfm_scores = np.load("user_item_lightfm_scores.npy")

    user_meta_data["user_lightfm_embeddings"] = user_embeddings.tolist()
    item_meta_data["item_lightfm_embeddings"] = item_embeddings.tolist()
    user_meta_data["user_lightfm_biases"] = user_biases.tolist()
    item_meta_data["item_lightfm_biases"] = item_biases.tolist()
    user_item_data["lightfm_scores"] = lightfm_scores.tolist()

    item_meta_data = item_meta_data.rename({"embeddings": "video_embeddings"}, axis=1)

    # np.uint8 -> np.int16 cast to allow subtraction
    user_item_data[user_item_data.dtypes[user_item_data.dtypes == np.uint8].index] = (
        user_item_data[
            user_item_data.dtypes[user_item_data.dtypes == np.uint8].index
        ].astype(np.int16)
    )

    # single column for likes and dislikes
    user_item_data["explicit"] = user_item_data.like - user_item_data.dislike

    user_item_data = user_item_data.merge(
        item_meta_data.drop(columns="video_embeddings"), on="item_id", how="left"
    )

    user_item_data = user_item_data.merge(user_meta_data, on="user_id", how="left")

    user_item_data["timespent_rel"] = (
        user_item_data["timespent"] / user_item_data["duration"]
    )

    train_df, val_df = train_test_split(user_item_data, test_size=0.2)

    ##### catboost
    columns = [
        "user_id",
        "item_id",
        "source_id",
        "duration",
        "gender",
        "age",
        "lightfm_scores",
    ]
    target = "explicit"

    train_df = train_df.sort_values(by="user_id", axis=0)
    val_df = val_df.sort_values(by="user_id", axis=0)

    train_pool = Pool(
        data=train_df[columns].values,
        label=train_df[target].values,
        group_id=train_df["user_id"].values.tolist(),
    )

    val_pool = Pool(
        data=val_df[columns].values,
        label=val_df[target].values,
        group_id=val_df["user_id"].values.tolist(),
    )

    del (
        train_df,
        val_df,
        user_item_data,
        user_meta_data,
        item_meta_data,
        user_embeddings,
        item_embeddings,
    )
    gc.collect()

    default_parameters = {
        "iterations": 20,
        "verbose": True,
        "random_seed": 0,
    }

    parameters = {}

    model = fit_model(
        "YetiRank",
        train_pool=train_pool,
        test_pool=val_pool,
    )

    model.save_model('ranker')