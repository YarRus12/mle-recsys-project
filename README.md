# Подготовка виртуальной машины

## Склонируйте репозиторий

Склонируйте репозиторий проекта:

```
git clone https://github.com/yandex-praktikum/mle-project-sprint-4-v001.git
```

## Активируйте виртуальное окружение

Используйте то же самое виртуальное окружение, что и созданное для работы с уроками. Если его не существует, то его следует создать.

Создать новое виртуальное окружение можно командой:

```
python3 -m venv .venv
```

После его инициализации следующей командой

```
. .venv/bin/activate
```

установите в него необходимые Python-пакеты следующей командой

```
pip install -r requirements.txt
```

### Скачайте файлы с данными

Для начала работы понадобится три файла с данными:
- [tracks.parquet](https://storage.yandexcloud.net/mle-data/ym/tracks.parquet)
- [catalog_names.parquet](https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet)
- [interactions.parquet](https://storage.yandexcloud.net/mle-data/ym/interactions.parquet)
 
Скачайте их в директорию локального репозитория. Для удобства вы можете воспользоваться командой wget:

```
wget https://storage.yandexcloud.net/mle-data/ym/tracks.parquet

wget https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet

wget https://storage.yandexcloud.net/mle-data/ym/interactions.parquet
```

## Запустите Jupyter Lab

Запустите Jupyter Lab в командной строке

```
jupyter lab --ip=0.0.0.0 --no-browser
```

# Расчёт рекомендаций

Код для выполнения первой части проекта находится в файле `recommendations.ipynb`. 
Изначально, это шаблон. Используйте его для выполнения первой части проекта.  

Комментарий студента: recommendations.ipynb содержит первый этап проекта с комментариями и выводами

# Сервис рекомендаций

Код сервиса рекомендаций находится в файле `recommendations_service.py`.

Для запуска сервиса рекомендаций достаточно запустить виртуальное окружение с установленными библиотеками и выполнить код

    uvicorn recommendations_service:app --host 0.0.0.0 --port 8000 --reload



# Инструкции для тестирования сервиса

Код для тестирования сервиса находится в файле `test_service.py`.

Для запуска тестирования необходимо создать файл .env
с содержимым  
AWS_ACCESS_KEY_ID=YCAJEa#######w2pH6ASixkVD1V6OqIw  
AWS_SECRET_ACCESS_KEY=YCNViAgYJ#######AXurxFt-5ZAAH_ZQauS37kGWk4od83K  
где нужно удалить знаки #######

и запустить скрипт test_service.py  
необходимые файлы подтянутся из S3, а результаты тетирования можно будет проверить в файле test_service.log