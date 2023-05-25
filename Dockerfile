FROM python:3.10

COPY Requirements.txt .

RUN pip install -r Requirements.txt

COPY . .

EXPOSE $PORT

CMD gunicorn --worker-class gevent --workers=2 --bind 0.0.0.0:$PORT app:app 
