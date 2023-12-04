FROM python:3.8.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["gateway.py", "proto.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]