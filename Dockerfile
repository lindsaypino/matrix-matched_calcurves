FROM python:3.9

ADD bin/* /app/
ADD requirements.txt /app/
WORKDIR /app

RUN pip install -v --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "/app/calculate-loq.py"]
