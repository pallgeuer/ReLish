# Stage 1: Build the required conda environment
FROM continuumio/miniconda3:22.11.1 as condainstall

ARG PYTHON=3.9
ARG CUDA_VERSION=11.6
ARG PYTORCH=1.12

RUN --mount=type=cache,target=/opt/conda/pkgs \
    conda create --copy -p /env python=$PYTHON && \
    conda install -p /env -c conda-forge -c pytorch cudatoolkit=$CUDA_VERSION pytorch=$PYTORCH torchvision && \
    conda run -p /env python -m pip install --no-cache-dir wandb matplotlib

RUN find -name '*.a' -delete && \
    find -name '__pycache__' -type d -exec rm -rf '{}' '+'

# Stage 2: Repackage into a small Ubuntu image
FROM ubuntu:22.04

ENV PATH="/relish:/env/bin:$PATH"
ENV DATASET_PATH=/Datasets

COPY --from=condainstall /env /env
COPY . /relish
# EOF
