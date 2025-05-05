FROM mambaorg/micromamba:2.0.8

USER root
USER $MAMBA_USER
COPY environment.yml /opt/environment.yml
RUN micromamba create -f /opt/environment.yml && micromamba clean --all --yes

ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=glacier-snow-cover-mapping
