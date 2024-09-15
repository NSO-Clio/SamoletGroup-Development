FROM python:3.11

WORKDIR /code

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./app ./app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "8"]
