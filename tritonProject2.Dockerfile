FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN python3 -m pip install --upgrade pip
WORKDIR /workspace
RUN git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git && \
cd TransformerEngine && \
export NVTE_FRAMEWORK=pytorch && \
pip install -e .
