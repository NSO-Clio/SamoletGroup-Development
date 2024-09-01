# web

This is a service with web pages/backend logic for the task.

The service may be run inside the docker container.

If you want to run it locally, install the required dependencies and do
```
python -m app
```
This will run the flask development server.

If you want to run inside docker, build the image
```
docker build -t aijc_web .
```
And run the container
```
docker run --rm -p 8000:80 aijc_web
```
