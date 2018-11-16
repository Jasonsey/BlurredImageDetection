FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL Name=blurred_image_detection Version=0.0.1
EXPOSE 9099
ENV TZ Asia/Shanghai
ENV APP_ROOT /app

# Fix Hash sum mismatch Bug
# ENV DEBIAN_FRONTEND noninteractive

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN rm -r /etc/apt/sources.list.d && \
#     mkdir -p /etc/apt/sources.list.d && \
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --allow-downgrades --allow-change-held-packages \
        libcudnn7=7.0.5.15-1+cuda9.0 \
        software-properties-common \
        thrift-compiler \
        python3.6 \
        libopencv-dev \
        python3-venv \
        wget && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_ROOT}

ADD requirements.txt ${APP_ROOT}
# Using pip:
RUN pip3.6 install \
        --no-cache-dir \
        --trusted-host 172.18.31.204 \
        -i http://172.18.31.204:8081/repository/pypi-aliyun/simple \
        -r requirements.txt

ADD . ${APP_ROOT}
# Prepare thrift
# RUN mkdir src/api/thrift_api && \
#     thrift -r -gen py -out src/api/thrift_api interface.thrift
# Fix thrift Bug
RUN sed -i "s/from ttypes/from .ttypes/g" src/api/thrift_api/interface/blur_detection.py

# Up Server
CMD ["make", "server"]

# For Deploy
# WORKDIR ${APP_ROOT}/src
# ENTRYPOINT ["python3.6", "main.py"]
# CMD ["server", "--gpu", "2"]
