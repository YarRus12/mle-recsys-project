import numpy as np
import pandas as pd
from scipy import sparse
import joblib
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Загрузка модели и кодировщиков
als_i2i_model = joblib.load('als_i2i_model.pkl')
events = pd.read_parquet('data/events.parquet')
data = pd.read_parquet('data/items.parquet')


# Функция для получения истории пользователя
def get_history(user_id, all_data):
    df = pd.read_parquet('data/events.parquet')
    df = df[df['user_id'] == user_id]
    last_two_months = df['started_at'].max() - pd.DateOffset(months=2)
    df = df[df['started_at'] > last_two_months]
    return df.merge(all_data, on='track_id', how='left')


def run(user_id):
    history_df = get_history(user_id=user_id, all_data=data)

    if history_df.empty:
        return {"error": "No history found for this user."}

    # Формируем список треков из истории
    tracks = history_df['track_id']
    encoder_track = joblib.load('encoder_track.pkl')

    item_factors = als_i2i_model.item_factors
    print(len(item_factors))
    track_ids = encoder_track.classes_
    print(len(track_ids))
    track_factors_df = pd.DataFrame(item_factors, index=track_ids)

    # Функция для получения похожих треков
    def get_similar_tracks(track_id, track_factors_df, n=5):
        if track_id not in track_factors_df.index:
            return []

        # Вычисляем схожесть между векторами факторов
        similarities = cosine_similarity(track_factors_df.loc[[track_id]], track_factors_df).flatten()
        # Получаем индексы наиболее схожих треков
        similar_indices = np.argsort(similarities)[::-1][1:n + 1]  # Исключаем сам трек
        # Возвращаем список похожих треков
        similar_tracks = track_factors_df.index[similar_indices].tolist()
        return similar_tracks

    results = []

    for track in tracks:
        similar_tracks = get_similar_tracks(track, track_factors_df)
        for similar_track in similar_tracks:
            results.append({'track_id': track, 'similar_track_id': similar_track})
    return results


# Пример вызова функции (для тестирования)
if __name__ == "__main__":
    print(run(592357))