FROM nvidia/cuda:9.0-devel

# Install basic tools
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y wget curl tar nano gedit ssh git python3-pip

RUN apt-get install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6

# Change the working directory
WORKDIR /repo

RUN pip3 install -U setuptools wheel && pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install matplotlib keras

ENTRYPOINT ["/bin/sh"]