FROM nvcr.io/nvidia/pytorch:23.10-py3
ARG CMAKE_VERSION=3.29.0
ARG NUM_JOBS=100
RUN python3 -m pip install --upgrade pip
WORKDIR /TransformerEngine
ENV NVTE_FRAMEWORK=pytorch
RUN git config --global --add safe.directory /TransformerEngine/TransformerEngine/3rdparty/cudnn-frontend
RUN git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git && \
    cd TransformerEngine && \
    pip install -e .

