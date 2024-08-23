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

ENV WORKDIR=/workdir
WORKDIR $WORKDIR/RoboCLIPv2

COPY ./roboclipv2.yml ./roboclipv2.yml 

# mujoco
RUN curl -o mjkey.txt https://www.roboti.us/file/mjkey.txt
RUN mkdir -p $WORKDIR/.mujoco
#RUN unzip mujoco210_linux.zip -d ~/.mujoco/mujoco210
#RUN rm mujoco210_linux.zip

RUN conda env create -f roboclipv2.yml
RUN mv mjkey.txt $WORKDIR/.mujoco/mjkey.txt
RUN wget -c https://www.roboti.us/download/mujoco200_linux.zip -O mujoco200.zip
RUN unzip mujoco200.zip -d $WORKDIR/.mujoco/
RUN rm mujoco200.zip
RUN mv $WORKDIR/.mujoco/mujoco200_linux $WORKDIR/.mujoco/mujoco200
RUN cd $WORKDIR && git clone https://github.com/avirupdas55/mujoco-py.git && cd mujoco-py && git switch v2.0.2.8-local_update 


# export the LD library path in the bashrc
ENV LD_LIBRARY_PATH $WORKDIR/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV MUJOCO_PY_MUJOCO_PATH ${WORKDIR}/.mujoco/mujoco200
ENV MUJOCO_PY_MJKEY_PATH ${WORKDIR}/.mujoco/mjkey.txt
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64" >> ~/.bashrc
#RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${WORKDIR}/.mujoco/mujoco200/bin" >> ~/.bashrc
#RUN echo "export MUJOCO_GL=egl" >> ~/.bashrc

# temp
#RUN pip install mujoco==2.2.1 mujoco-py==2.1.2.14 

SHELL ["conda", "run", "-n", "roboclip", "/bin/bash", "-c"]
RUN conda init bash

ENV D4RL_SUPPRESS_IMPORT_ERROR=1
RUN pip install -U Cython==3.0.0a10
RUN git clone https://github.com/sumedh7/kitchen_alt.git
RUN git clone https://github.com/sumedh7/Metaworld.git
RUN git clone https://github.com/aravindr93/mjrl.git

RUN pip install -e mjrl
RUN pip install -e Metaworld
RUN pip install -e kitchen_alt
RUN pip install -e kitchen_alt/kitchen/envs

RUN cd $WORKDIR && pip install -e mujoco-py
RUN python -c "import mujoco_py"
RUN apt-get install libc6 -y
RUN pip install opencv-python-headless

# COPY ./docker/entrypoint.sh ./docker/entrypoint.sh 
# RUN chmod +x ./docker/entrypoint.sh
# ENTRYPOINT ["docker/entrypoint.sh"]


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