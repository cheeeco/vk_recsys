from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

# Настройки для кластеризации
N_CLUSTERS = 10  # Количество кластеров, можно подбирать

def cluster_embeddings(item_meta_data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Кластеризация эмбеддингов и добавление категориальной фичи imageCat.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    embeddings = np.vstack(item_meta_data['embeddings'])
    item_meta_data['imageCat'] = kmeans.fit_predict(embeddings)
    return item_meta_data

def generate_user_features(
    user_item_data: pd.DataFrame, item_meta_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Генерация новых категориальных фич и счетчиков на основе кластеров.
    """
    # Объединение данных
    merged_data = user_item_data.merge(item_meta_data[['item_id', 'imageCat']], on='item_id', how='left')
    
    # Генерация фич
    user_features = merged_data.groupby('user_id').apply(generate_features_for_user).reset_index(drop=True)
    
    return user_features

def generate_features_for_user(user_data: pd.DataFrame) -> pd.Series:
    """
    Генерация фич для конкретного пользователя.
    """
    user_id = user_data['user_id'].iloc[0]
    image_cat_counts = Counter(user_data['imageCat'])
    
    # Категориальные фичи
    most_watched_image_cat = max(image_cat_counts, key=image_cat_counts.get)
    liked_image_cats = user_data[user_data['like'] == 1]['imageCat']
    most_liked_image_cat = liked_image_cats.value_counts().idxmax() if not liked_image_cats.empty else -1
    
    # Счетчики
    unique_image_cats_count = len(image_cat_counts)
    total_items_watched = len(user_data)
    
    return pd.Series({
        'user_id': user_id,
        'most_watched_imageCat': most_watched_image_cat,
        'most_liked_imageCat': most_liked_image_cat,
        'unique_imageCats_count': unique_image_cats_count,
        'total_items_watched': total_items_watched,
    })

def main(user_item_data: pd.DataFrame, item_meta_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная функция: кластеризация эмбеддингов и генерация фич.
    """
    # Кластеризация эмбеддингов
    item_meta_data = cluster_embeddings(item_meta_data, N_CLUSTERS)
    
    # Генерация фич
    user_features = generate_user_features(user_item_data, item_meta_data)
    
    return item_meta_data, user_features

if __name__ == "__main__":
    # Пример использования
    user_item_data = pd.read_csv('user_item_data.csv')  # Замените на ваш путь
    item_meta_data = pd.read_csv('item_meta_data.csv')
    
    # Преобразование строки с эмбеддингами в numpy array
    item_meta_data['embeddings'] = item_meta_data['embeddings'].apply(eval).apply(np.array)
    
    # Генерация фич
    item_meta_data, user_features = main(user_item_data, item_meta_data)
    
    # Сохранение результатов
    user_features.to_csv('user_features.csv', index=False)
    item_meta_data.to_csv('item_meta_data_with_clusters.csv', index=False)
