FROM tensorflow/tensorflow:latest-gpu
LABEL maintainer="prange@informatik.uni-freiburg.de"
ENV PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y make vim python3-pip git wget default-jre

# Install python packages
# For benepar setup, cython and numpy need to be installed separately
RUN pip3 install Cython==0.29.12
RUN pip3 install numpy==1.17.2

COPY requirements.txt /home/requirements.txt
RUN pip3 install -r /home/requirements.txt

# Download spaCy models
RUN python3 -m spacy download de_core_news_sm
RUN python3 -m spacy download en_core_web_sm
RUN python3 -m spacy download en

# Create directory structure
RUN mkdir -p /home/FQG/src/model/FactorizedQG/

# Copy code to code path
COPY . /home/FQG/src/model/FactorizedQG/

WORKDIR /home/FQG/src/model/FactorizedQG/

# Download benepar model
RUN python3 /home/FQG/src/model/FactorizedQG/setup.py

# docker build -t acs-qg-docker .
# docker run -it -v /nfs/students/natalie-prange/ACS-QG_data:/home/Datasets -v /nfs/students/natalie-prange/ACS-QG_data/output:/home/FQG/output acs-qg-docker
