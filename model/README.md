# Model

Это - микросервис, который может быть использован, чтобы легко разворачивать инстансы моделей.
Доступны и докеризированная, и обычная версия, на локальной машине.

В обоих случаях необходимо установить переменные серды с путём до весом модели. Если запускать модель на локальной машине, поставить это можно с


```
export SCORING_MODEL_PATH="/path_to_model" SCORING_CBC_PATH="/path_to_catboost_model" SCORING_XGBC_PATH="/path_to_xgb_model" SCORING_MEAN_MEDIAN_IMPUTER_PATH="/path_to_mmimputer"
```
Также можно создать .env файл и прописать это там.

Для докера необходимо поставить так:
```
docker run -e SCORING_MODEL_PATH="/weights/model" -e SCORING_CBC_PATH="/path_to_catboost_model" -e SCORING_XGBC_PATH="/path_to_xgb_model" -e SCORING_MEAN_MEDIAN_IMPUTER_PATH="/weights/mmimputer" -v ./weights:/weights ...
```

## Docker версия

Необходимо собрать докер образ с
```
docker build -t aij_model .
```
И запустить его:
```
docker run -e SCORING_MODEL_PATH="/weights/model" \
	-e SCORING_CBC_PATH="/path_to_catboost_model" \
	-e SCORING_XGBC_PATH="/path_to_xgb_model" \
	-e SCORING_MEAN_MEDIAN_IMPUTER_PATH="/weights/mmimputer" \
	-v ./weights:/weights \
	-p 8000:8000 \
	aij_model
```

## Запуск на локальной машине

Необходимо установить пакеты из requirements.txt в оболочку с python 3.11 и запустить сервер для разработчика через
```
fastapi dev app
```
