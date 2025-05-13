FROM mambaorg/micromamba:latest

# Set working directory
WORKDIR /app

# Cop files into container
COPY scripts .
COPY notebooks .
COPY functions .
COPY inputs-outputs .

# Copy yaml file into the container
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install dependencies using micromamba
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes