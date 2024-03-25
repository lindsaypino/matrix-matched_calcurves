FROM python:3.9

# Set environment variables for Matplotlib and Fontconfig
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV HOME=/tmp/home

ADD bin/* /app/
ADD requirements.txt /app/
WORKDIR /app

RUN pip install -v --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "/app/calculate-loq.py"]
