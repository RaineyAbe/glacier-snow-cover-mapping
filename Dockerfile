FROM mambaorg/micromamba:latest

# Set working directory
WORKDIR /app

# Copy yaml file into the container
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install dependencies using micromamba
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes