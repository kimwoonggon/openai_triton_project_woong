FROM nvcr.io/nvidia/pytorch:23.10-py3

ARG CMAKE_VERSION=3.27.3
ARG NUM_JOBS=100

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        libjpeg-dev \
        libpng-dev \
        language-pack-en \
        locales \
        locales-all \
        libprotobuf-dev \
        protobuf-compiler \
        zlib1g-dev \
        swig \
        vim \
        gdb \
        valgrind \
        libsm6 \
        libxext6 \
        libxrender-dev \
        cmake \
        unzip \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        libavresample-dev \
        ffmpeg \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        libopenexr-dev \
        libtiff-dev \
        libwebp-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@stable
