FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
