FROM ai-container-base

ARG USERNAME=trist
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN git clone https://github.com/openai/triton.git /home/$USERNAME/triton \
    && git clone https://github.com/tristanheywood/jax-triton /home/$USERNAME/jax-triton \
    && git clone https://github.com/tristanheywood/nimbleGPT /home/$USERNAME/nimbleGPT \
    && git clone https://github.com/tristanheywood/ai-notebooks.git /home/$USERNAME/notebooks

# Install Triton
WORKDIR /home/$USERNAME/triton/python
RUN pip install cmake && PATH="/home/$USERNAME/.local/bin/:$PATH" pip install -e .

# Install jax-triton (not working)
# WORKDIR /home/$USERNAME/jax-triton
# RUN pip install -e .