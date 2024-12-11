from feature_processor import create_features
from utils import load_data
import pandas as pd

# user_item_data, user_meta_data, item_meta_data, test_pairs_data = load_data()
# user_item_data = user_item_data.merge(
#     right=item_meta_data.drop(columns="embeddings"),
#     on="item_id",
#     how="left",
# )
# test_pairs_data = test_pairs_data.merge(
#     right=item_meta_data.drop(columns="embeddings"),
#     on="item_id",
#     how="left",
# )
# user_item_data = user_item_data.merge(
#     right=user_meta_data,
#     on="user_id",
#     how="left",
# )
# test_pairs_data = test_pairs_data.merge(
#     right=user_meta_data,
#     on="user_id",
#     how="left",
# )

# # history_df = user_item_data[:-int(len(user_item_data) * 0.15)]
# # val_df = user_item_data[-int(len(user_item_data) * 0.15) :]

# val_df, _ = create_features(history_df=user_item_data,
#                             train_df=None,
#                             val_df=test_pairs_data)

# print(val_df[-5:])
# del history_df, user_item_data, test_pairs_data, val_df, user_meta_data

user_item_data_w_group_features = pd.read_parquet("data/test_pairs_data_w_group_features.parquet")
print(user_item_data_w_group_features[-5:])

