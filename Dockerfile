# base image
FROM python:3.11-slim
# prevents python from writing .pyc files and enables unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1
# set the working directory
WORKDIR /app
# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*
# install pip
RUN pip install --upgrade pip
# copying requirements.txt first for better caching
COPY requirements.txt .
# install dependencies
RUN pip install -r requirements.txt
# copy rest of the project
COPY . .
# expose FastAPI port
EXPOSE 5000
# run FastAPI with uvicorn
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5000"]
