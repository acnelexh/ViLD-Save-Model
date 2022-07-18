FROM tensorflow/tensorflow:latest-gpu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  git \
  ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*
  
RUN pip3 install --upgrade pip
RUN pip3 install PyYAML
RUN pip3 install easydict
RUN pip3 install scipy
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 uninstall -y torch torchvision
RUN pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111