FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim
ENV DEBIAN_FRONTEND=noninteractive
# lib requirements
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3-opencv \
    libeccodes-dev \
    software-properties-common \
    build-essential \
    libssl-dev \
  && apt-get autoremove -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install gribflow
COPY . /app
RUN python3 /app/setup.py install
# re-install opencv-contrib-python
RUN pip uninstall -y opencv-contrib-python && \
    pip install --no-cache opencv-contrib-python==4.5.3.56
# expects fastapi as main.py in /app
RUN cp /app/gribflow/app.py /app/main.py