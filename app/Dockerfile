FROM  tensorflow/tensorflow:2.11.0

WORKDIR /app

COPY app/requirements.txt ./
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

COPY app/api.py ./
COPY params.yaml ./
COPY models/model.tf ./models/model.tf
COPY models/score.json ./models/score.json

EXPOSE 5000

CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000" ]
