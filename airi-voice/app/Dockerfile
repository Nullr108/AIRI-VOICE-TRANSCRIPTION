FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Базовые зависимости
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg git libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем зависимости (coqui-tts для XTTS-v2)
RUN pip3 install fastapi uvicorn[standard] faster-whisper coqui-tts soundfile python-multipart

# Проверяем версию CUDA
RUN nvcc --version

# Устанавливаем cuDNN
RUN apt-get update && \
    apt-get install -y wget gnupg2 software-properties-common && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb && \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor > /usr/share/keyrings/cuda-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | tee /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y libcudnn8 libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*

# Проверяем версию cuDNN
RUN dpkg -l | grep libcudnn

WORKDIR /app
COPY app.py /app
COPY sample.wav /app

CMD ["bash", "-c", "yes y | uvicorn app:app --host 0.0.0.0 --port 8000"]
