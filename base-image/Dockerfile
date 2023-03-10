# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
# https://github.com/orgs/pytorch/packages/container/pytorch-nightly/74629833?tag=5bdd054-cu11.7.0
# FROM ghcr.io/pytorch/pytorch-nightly:5bdd054-cu11.7.0
FROM ghcr.io/pytorch/pytorch-nightly:2.1.0.dev20230303-devel

# Install gcc/g++ v9 for building triton.
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get -y install gcc-9 g++-9 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9

RUN apt-get install -y vim git byobu htop

ARG USERNAME=trist
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user
# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
# End Copy

# Needed to let conda install packages.
RUN sudo chown $USERNAME -R /opt/conda

COPY bashrc /home/$USERNAME/.bashrc
# Make byobu use bash as its shell (doens't work - maybe this file doens't exist until byobu is run).
COPY tmux.conf /home/$USERNAME/.byobu/.tmux.conf
# RUN echo "set -g default-shell /bin/bash\nset -g default-command /bin/bash" >> /home/$USERNAME/.byobu/.tmux.conf

# Start the container in the home directory.
WORKDIR /home/$USERNAME

# This is where cmake will be installed.
ENV PATH="${PATH}:/home/$USERNAME/.local/bin"

RUN pip install cmake \
 && pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Nsight systems for profiling
RUN sh -c sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'
# Fails since moving to new Pytorch 2.0 image.
RUN sudo apt-get update && \
    sudo apt-get install -y cuda-nsight-systems-11-6 nsight-systems-2022.4.2