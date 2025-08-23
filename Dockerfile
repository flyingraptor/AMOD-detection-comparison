FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# default python ver == 3.10
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

    # RUN apt-get update && apt-get install -y software-properties-common && \
    # add-apt-repository ppa:deadsnakes/ppa && \
    # apt-get update && apt-get install -y \
    # python3.8 \
    # python3.8-dev \
    # python3.8-distutils \
    # python3-pip \
    # git \
    # && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install -U openmim==0.3.9
RUN mim install mmcv-full==1.7.2

COPY . .

RUN pip3 install -v -e mmdetection/
RUN pip3 install -v -e mmrotate/

CMD ["/bin/bash"]

# 내가 빌드한 도커 이미지 공유하기!!!
# 1-이미지에 태그 붙이기 (tag)
# 형식: docker tag <local-image>:<tag> <DockerHub-username>/<repo>:<tag>
# docker tag my-amod-app:v1 uniquechan/my-amod-app:v1
#
# 2-로그인
# docker login
#
# 3-도커 이미지 클라우드 허브에 업로드하기 (push)
# docker push uniquechan/my-amod-app:v1