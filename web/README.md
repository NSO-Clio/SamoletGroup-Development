# web

This is a service with web pages/backend logic for the task.

Before run it you should also set `FLASK_MODEL_HOST` environment variable
with the model destination host (like http://127.0.0.1:8000)


The service may be run inside the docker container.

If you want to run it locally, install the required dependencies and do
```
FLASK_MODEL_HOST="http://127.0.0.1:8000" python -m app
```
Or use .env file with that environment.

If you want to run inside docker, build the image
```
docker build -t aijc_web .
```
And run the container
```
docker run --rm \
	-p 8000:80 \
	-e FLASK_MODEL_HOST="http://your_model:8000" 
	aijc_web
```

Note, that for docker you should forward port with your model to the web. This may be done via, for example, docker networks.
