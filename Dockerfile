FROM ubuntu:focal

# Install basic tools
RUN apt update && DEBIAN_FRONTEND=noninteractive apt install -y wget curl tar nano gedit ssh git python3-pip openjdk-8-jdk supervisor

# Change the working directory
WORKDIR /

RUN pip3 install -U setuptools wheel && pip3 install torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install matplotlib
