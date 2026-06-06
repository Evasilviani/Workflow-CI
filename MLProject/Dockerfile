FROM python:3.10-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY heart_preprocessing.csv .

RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    scikit-learn==1.4.2 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    dagshub==0.3.36 \
    cloudpickle

CMD ["python", "modelling.py"]
