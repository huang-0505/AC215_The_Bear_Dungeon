# Use an appropriate base image
FROM continuumio/miniconda3:latest

# Set the working directory first to reduce invalidation of layers in case files in /app change.
WORKDIR /app

# Copy the conda environment file
COPY environment_droplet.yml /environment_droplet.yml

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Create conda environment and clean up
RUN conda env create -n venv -f /environment_droplet.yml && \
    conda clean -a

RUN echo "source activate venv" > ~/.bashrc

# Activate the conda environment
ENV PATH /opt/conda/envs/venv/bin:$PATH

# RUN conda activate venv

# Create a directory for data
RUN mkdir /app/data

# Uncomment and adjust as necessary
# COPY tidy-resolver-411707-0f032726c297.json /app/tidy-resolver-411707-0f032726c297.json
# COPY trainer.py /app/trainer.py

# Set the default entrypoint
ENTRYPOINT  ["/bin/bash"]

