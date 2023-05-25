<<<<<<< HEAD
FROM python:3.10

COPY Requirements.txt .

RUN pip install -r Requirements.txt

COPY . .

EXPOSE $PORT

CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app

=======
FROM python:3.10-alpine

COPY Requirements.txt .

RUN pip install -r Requirements.txt

COPY . .

EXPOSE $PORT

CMD gunicorn --workers=2 --bind 0.0.0.0:$PORT app:app

>>>>>>> c38fa9baf9fe26ef02b6aeebfd607ddb8ee2fa95
