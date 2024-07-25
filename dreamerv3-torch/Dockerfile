# Base image with PyTorch, CUDA, and cuDNN
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set non-interactive installation mode and timezone to avoid prompts
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    vim libglew2.1 libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    && apt-get clean

# Upgrade pip
RUN pip3 install --upgrade pip

# Set the NUMBA cache directory
ENV NUMBA_CACHE_DIR=/tmp

# Install Python libraries for various setups
RUN pip3 install \
    tensorboard \
    gym==0.19.0 \
    mujoco==2.3.5 \
    dm_control==1.0.9 \
    moviepy \
    crafter==1.8.0 \
    atari-py==0.2.9 \
    opencv-python==4.7.0.72 \
    memory_maze==1.0.3 \
    numpy

# Preparation steps for Atari setup (e.g., downloading ROMs) are removed
# to adhere to the request of not running any code automatically.

# Setting up a working directory
WORKDIR /workspace

# Command to start an interactive bash shell when the container starts
CMD ["/bin/bash"]
