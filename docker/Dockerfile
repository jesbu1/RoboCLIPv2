# FROM nvidia/cuda:12.5.1-devel-ubuntu20.04
#FROM nvcr.io/nvidia/pytorch:23.03-py3
FROM nvcr.io/nvidia/pytorch:21.09-py3

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update

WORKDIR /home/root/RoboCLIPv2

COPY ./roboclipv2.yml ./roboclipv2.yml 

# mujoco
RUN curl -o mjkey.txt https://www.roboti.us/file/mjkey.txt
RUN mkdir -p ~/.mujoco
RUN mv mjkey.txt ~/.mujoco/mjkey.txt
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
#RUN unzip mujoco210_linux.zip -d ~/.mujoco/mujoco210
#RUN rm mujoco210_linux.zip
RUN rm mujoco210-linux-x86_64.tar.gz

RUN conda env create -f roboclipv2.yml

# export the LD library path in the bashrc
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco200/bin" >> ~/.bashrc
#RUN echo "export MUJOCO_GL=egl" >> ~/.bashrc

# temp
#RUN pip install mujoco==2.2.1 mujoco-py==2.1.2.14 

SHELL ["conda", "run", "-n", "roboclip", "/bin/bash", "-c"]
RUN conda init bash

COPY ./docker/entrypoint.sh ./docker/entrypoint.sh 
ENTRYPOINT ["docker/entrypoint.sh"]


# RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
#     && chmod +x /usr/local/bin/patchelf

# ENV LANG C.UTF-8

# RUN mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# COPY vendor/Xdummy /usr/local/bin/Xdummy
# RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
#COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

#WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
#COPY ./requirements.txt /mujoco_py/
#COPY ./requirements.dev.txt /mujoco_py/
#RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
#ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
#CMD ["pytest"]
#COPY . /mujoco_py
#RUN python setup.py install