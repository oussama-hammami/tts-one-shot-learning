FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

COPY . /app

RUN apt-get update && \
    apt-get install -y --allow-unauthenticated --no-install-recommends \
    wget \
    git \
    build-essential \
    gcc \
    g++ \
    libaio-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV HOME="/root"
ENV CONDA_DIR="${HOME}/miniconda"
ENV PATH="$CONDA_DIR/bin":$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PIP_DOWNLOAD_CACHE="$HOME/.pip/cache"
ENV TORTOISE_MODELS_DIR="$HOME/tortoise-tts/build/lib/tortoise/models"

# Set CUDA architecture for compilation
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh \
    && echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

# --login option used to source bashrc (thus activating conda env) at every RUN statement
SHELL ["/bin/bash", "--login", "-c"]

# Accept Anaconda TOS before creating environments
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create --name tortoise python=3.9 numba inflect -y \
    && conda activate tortoise \
    && conda install --yes pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia \
    && pip install deepspeed==0.9.5 \
    && conda install --yes transformers=4.29.2 \
    && pip install --upgrade pip setuptools wheel \
    && cd /app \
    && pip install -r requirements.txt \
    && python setup.py install \
    && conda clean -afy \
    && find /root/miniconda -follow -type f -name '*.a' -delete \
    && find /root/miniconda -follow -type f -name '*.pyc' -delete \
    && find /root/miniconda -follow -type f -name '*.js.map' -delete \
    && pip cache purge

# Create output directory for generated files
RUN mkdir -p /app/output

WORKDIR /app
CMD ["bash", "-c", "source activate tortoise && python tts_test.py"]