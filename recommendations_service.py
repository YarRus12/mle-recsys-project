import logging
import pandas as pd
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI
import boto3
from dotenv import load_dotenv
import os
import io

load_dotenv()

BUCKET_NAME = 's3-student-mle-20240822-03e9c191e2'
ENDPOINT = "https://storage.yandexcloud.net"
aws_access_key_id = f"{os.getenv('AWS_ACCESS_KEY_ID')}"
aws_secret_access_key = f"{os.getenv('AWS_SECRET_ACCESS_KEY')}"

s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        verify=False,
    )

logger = logging.getLogger("uvicorn.error")


def get_data_s3(path):

    response = s3.get_object(Bucket=BUCKET_NAME, Key=path)
    buffer = io.BytesIO(response['Body'].read())
    df = pd.read_parquet(buffer)
    return df


class Recommendations:

    def __init__(self):

        self._recs = {"personal": None, "online": None, "default": None}
        self._stats = {
            "request_personal_count": 0,
            "request_default_count": 0,
        }

    async def load(self, type_rec, path, **kwargs):
        """
        Загружает рекомендации из файла
        """
        logger.info(f"Loading recommendations, type: {type_rec}")
        self._recs[type_rec] = get_data_s3(path)
        self._recs[type_rec] = self._recs[type_rec]
        logger.info(f"Loaded")

    async def get(self, user_id: int, k: int = 100):
        """
        Возвращает список рекомендаций для пользователя
        """
        recs = {
            "personal": [],
            "default": []
        }

        personal_recs = self._recs["personal"][self._recs["personal"]["user_id"] == user_id]
        recs["personal"] = personal_recs["track_id"].to_list()[:k]

        # Если нет персональных рекомендаций - возвращает рекомендации из топ 100
        if len(recs["personal"]) == 0:
            logger.info('No personal')
            await self.load(type_rec="default",
                      path="recsys/recommendations/top_popular.parquet",
                      columns=["track_id"],
                      )
            default_recs = self._recs["default"]
            recs["default"] = default_recs["track_id"].to_list()[:k]

        return recs

    def stats(self):

        logger.info("Stats for recommendations")
        for name, value in self._stats.items():
            logger.info(f"{name:<30} {value} ")

        logger.info("Done")


class SimilarItems:

    def __init__(self):

        self._similar_items = None

    def load(self, path, type="online", **kwargs):
        """
        Загружаем данные из файла
        """

        logger.info(f"Loading data, type: {type}")
        self._similar_items = pd.read_parquet(path)
        logger.info(f"Loaded")

    def get(self, item_id: int, k: int = 10):
        """
        Возвращает список похожих объектов
        """
        try:
            i2i = self._similar_items[self._similar_items["track_id"] == item_id].head(k)
            i2i = i2i[["similar_track_id"]].to_dict(orient="list")
        except KeyError as e:
            logger.error(e)
            i2i = {"track_id": [], "similar_track_id": []}

        return i2i


class EventStore:

    def __init__(self, max_events_per_user=10):
        self.events = {}
        self.max_events_per_user = max_events_per_user

    def put(self, user_id: int, item_id: int):
        """
        Сохраняет событие.
        Если пользователь уже имеет события, добавляем новое событие в начало списка.
        Если количество событий превышает максимальное значение, удаляем старые.
        """
        if user_id not in self.events:
            self.events[user_id] = []

        # Добавляем новое событие
        user_events = self.events[user_id]
        user_events.insert(0, item_id)  # Добавляем новое событие в начало списка

        # Ограничиваем количество событий до max_events_per_user
        self.events[user_id] = user_events[:self.max_events_per_user]

    def get(self, user_id: int, k: int) -> List[int]:
        """
        Возвращает события для пользователя.
        Если пользователь не имеет событий, возвращаем пустой список.
        """
        return self.events.get(user_id, [])[:k]


@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_items_store.load(
        type_rec="online",
        path="data/similar.parquet",
        columns=["track_id", "similar_track_id"],
    )
    logger.info("Ready!")
    yield


app = FastAPI(title="features", lifespan=lifespan)
events_store = EventStore()
sim_items_store = SimilarItems()


@app.post("/similar_items")
async def online_recommendations(item_id: int, k: int = 10):
    """
    Возвращает список похожих объектов длиной k для item_id
    """
    i2i = sim_items_store.get(item_id, k)
    return i2i


@app.post("/put")
async def put(user_id: int, item_id: int):
    """
    Сохраняет событие для user_id, item_id
    """

    events_store.put(user_id, item_id)

    return {"result": "ok"}


@app.post("/get")
async def get(user_id: int, k: int = 10):
    """
    Возвращает список последних k событий для пользователя user_id
    """

    events = events_store.get(user_id, k)

    return {"events": events}


def dedup_ids(ids):
    """
    Дедублицирует список идентификаторов, оставляя только первое вхождение
    """
    seen = set()
    ids = [id for id in ids if not (id in seen or seen.add(id))]

    return ids


@app.post("/recommendations_online")
async def recommendations_online(user_id: int, k: int = 100):
    """
    Возвращает список онлайн-рекомендаций длиной k для пользователя user_id
    """
    # получаем последнее событие пользователя
    resp = await get(user_id, k=k)
    events = resp["events"]
    results = []
    # получаем список похожих объектов
    if len(events) > 0:
        for event in events:
            result = await online_recommendations(item_id=event, k=k)
            results += result.get('similar_track_id')
    return {"recs": results}


@app.post("/recommendations_offline")
async def recommendations_offline(user_id: int, k: int = 100):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """
    rec_store = Recommendations()
    await rec_store.load(type_rec="personal",
                   path="recsys/recommendations/personal_als.parquet",
                   columns=["user_id", "track_id"],
                   )
    recs = await rec_store.get(user_id, k)
    flat_recs = [track for sublist in recs.values() for track in sublist]
    return {"recs": flat_recs}


@app.post("/recommendations")
async def recommendations(user_id: int, k: int = 200):
    """
    Возвращает список рекомендаций длиной k для пользователя user_id
    """

    recs_offline = await recommendations_offline(user_id, k)
    recs_online = await recommendations_online(user_id, k)

    recs_offline = recs_offline["recs"]
    recs_online = recs_online["recs"]

    recs_blended = []

    min_length = min(len(recs_offline), len(recs_online))
    # чередуем элементы из списков, пока позволяет минимальная длина
    for i in range(min_length):
        recs_blended.append(recs_online[i])
        recs_blended.append(recs_offline[i])

    # добавляем оставшиеся элементы в конец
    recs_blended += recs_online[min_length:]
    recs_blended += recs_offline[min_length:]

    # удаляем дубликаты
    recs_blended = dedup_ids(recs_blended)

    return {"recs": recs_blended[:k]}
