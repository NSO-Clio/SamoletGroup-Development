FROM python:3.11

WORKDIR /code

RUN pip install uwsgi

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app ./app


COPY uwsgi.ini .

CMD [ "uwsgi", "uwsgi.ini" ]
