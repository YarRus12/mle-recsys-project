import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Загрузка модели и кодировщиков
als_i2i_model = joblib.load('als_i2i_model.pkl')
als_model = joblib.load('als_model.pkl')
events = pd.read_parquet('data/events.parquet')
data = pd.read_parquet('data/items.parquet')
merged = events.merge(data, on='track_id', how='left')


def get_top_contents() -> list:
    df = pd.read_parquet('data/top_popular.parquet')
    top_tracks = df['track_id'].unique()
    return top_tracks


# Функция для получения истории пользователя
def get_history(user_id, all_data):
    df = pd.read_parquet('data/events.parquet')
    df = df[df['user_id'] == user_id]
    last_two_months = df['started_at'].max() - pd.DateOffset(months=2)
    df = df[df['started_at'] > last_two_months]
    return df.merge(all_data, on='track_id', how='left')


def get_sparse_matrix_csr():
    # Создание разреженной матрицы взаимодействия
    interaction_matrix = merged.groupby(['user_id', 'track_id']).size().unstack(fill_value=0)

    # Преобразование индексов в NumPy массивы
    user_ids = interaction_matrix.index.to_numpy()
    track_ids = interaction_matrix.columns.to_numpy()

    # Создание отображения для преобразования идентификаторов в индексы
    user_id_to_index = {id: index for index, id in enumerate(user_ids)}
    track_id_to_index = {id: index for index, id in enumerate(track_ids)}

    # Создание разреженной матрицы взаимодействия
    rows = []
    cols = []
    data = []

    for (user_id, track_id), count in interaction_matrix.stack().items():
        rows.append(user_id_to_index[user_id])
        cols.append(track_id_to_index[track_id])
        data.append(count)

    # Преобразование в NumPy массивы
    rows_np = np.array(rows, dtype=np.int32)
    cols_np = np.array(cols, dtype=np.int32)
    data_np = np.array(data, dtype=np.float32)

    # Создание разреженной матрицы в формате COO
    sparse_matrix = sparse.coo_matrix((data_np, (rows_np, cols_np)),
                                      shape=(len(user_ids), len(track_ids)))

    # Преобразование в формат CSR для эффективной индексации
    sparse_matrix_csr = sparse_matrix.tocsr()
    return sparse_matrix_csr, track_ids


def personal_recommendations(user_id, model_personal, history_df) -> list:
    # Создание списка для хранения рекомендаций и прослушанных значений
    user_id_to_index = {id: index for index, id in enumerate([user_id])}
    recommendations = []
    true_labels = []

    sparse_matrix_csr, track_ids = get_sparse_matrix_csr()
    user_index = user_id_to_index[user_id]
    recommended = model_personal.recommend(user_index, sparse_matrix_csr[user_index], N=10)

    # Извлекаем индексы треков и их оценки
    track_indices, scores = recommended

    # Сохраняем рекомендации
    for track_index in track_indices:
        recommendations.append(
            (user_id, track_ids[track_index]))  # track_ids[track_index] возвращает идентификатор трека

    # Сохраняем истинные значения (треки, которые пользователь прослушал)
    true_tracks = [history_df['user_id'] == user_id]['track_id'].values
    true_labels.extend([(user_id, track) for track in true_tracks])

    # Создание DataFrame с рекомендациями
    recommendations_df = pd.DataFrame(recommendations, columns=['user_id', 'track_id'])

    return recommendations_df['track_id'].to_list()


def online_recommendations(history_df, model):
    # Формируем список треков из истории пользователя
    tracks = history_df['track_id']
    encoder_track = joblib.load('encoder_track.pkl')
    track_ids = encoder_track.classes_
    item_factors = model.item_factors

    def get_similar_tracks(track_id, track_factors_df, n=5):
        # Функция для получения похожих треков
        if track_id not in track_factors_df.index:
            return []
        # Вычисляем схожесть между векторами факторов
        similarities = cosine_similarity(track_factors_df.loc[[track_id]], track_factors_df).flatten()
        # Получаем индексы наиболее схожих треков
        similar_indices = np.argsort(similarities)[::-1][1:n + 1]  # Исключаем сам трек
        # Возвращаем список похожих треков
        similar_tracks = track_factors_df.index[similar_indices].tolist()
        return similar_tracks

    results = [
        similar_track
        for track in tracks
        for similar_track in get_similar_tracks(track, pd.DataFrame(item_factors, index=track_ids))
    ]
    return results


@app.get("/recommendations/{user_id}")
def run(user_id):
    history_df = get_history(user_id=user_id, all_data=data)
    if history_df.empty:
        print(f'Новый пользователь')
        result = get_top_contents()
        print(len(result))
        return result
    elif history_df['track_id'].nunique() < 50:
        print(f'Формируем персональные рекомендации')
        results = personal_recommendations(user_id, als_model)
        return results
    else:
        print(f'Формируем персональные и онлайн рекомендации')
        online = online_recommendations(user_id, history_df, als_i2i_model)
        personal = personal_recommendations(user_id, als_model)
        results = online + personal
    return results


# Пример вызова функции (для тестирования)
if __name__ == "__main__":
    print(run(user_id=592357592357))
    #print(run(592357))
