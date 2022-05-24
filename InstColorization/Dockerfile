FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY env.yml .
RUN conda env create -f env.yml

# Activate the environment, and make sure it's activated:
RUN conda init bash

RUN echo "conda activate instacolorization" > ~/.bashrc
RUN echo "Make sure flask is installed:"
RUN python -c "import os"

# The code to run when container is started:
COPY download.py .
ENTRYPOINT ["python", "download.py"]