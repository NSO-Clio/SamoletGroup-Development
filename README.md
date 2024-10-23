# Development AIIJC

Решение задачи Development хакатона AI Challenge команды CLIO.

Этот репозиторий предоставляет код нашего веб-сервиса.

Немного про архитектуру: модель вынесена в отдельный микросервис для масштабируемости, в дальнейшем можно будет создавать пул моделей и выносить их за балансировщик нагрузки, например, nginx.

Базовый запуск в production mode можно сделать через docker compose, находясь в корне проекта:

```sh
docker compose up --build
```

Следует отметить, что если запускать так, в папке model/weights должны находиться веса моделей.

Чтобы запустить development версии серверов, необходимо вручную создать окружение, прописать переменные среды в .env файлах и независимо запустить команды старта сервера (подробнее - в README к model/ и web/)

