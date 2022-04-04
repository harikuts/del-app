# For more information, please refer to https://aka.ms/vscode-docker-python
FROM debian:buster-slim
# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# # Get Linux image. (modified from the default VSCODE Python Dockerfile generation.)
# FROM debian:buster-slim
# # Non-interactive to prevent user input prompts.
# ARG DEBIAN_FRONTEND=noninteractive

# # Install essential binaries.
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git python3-dev python3-pip
# # Get network utilities.
# RUN apt-get install -y bridge-utils iperf3

# Install pip requirements.
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install -r requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /app
COPY . /app

ENV PYTHONPATH /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD python3 init.py ; python3 train.py & python3 aggregate.py & python3 rx.py & python3 tx.py
# CMD python3 learn.py