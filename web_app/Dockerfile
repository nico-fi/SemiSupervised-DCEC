FROM python:3-slim

WORKDIR /usr/src/app

COPY web_app/requirements.txt ./
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

COPY web_app/web_app.py ./
COPY data/samples ./data/samples

EXPOSE 4000

CMD [ "python", "./web_app.py" ]
