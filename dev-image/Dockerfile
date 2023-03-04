FROM ai-container-base

ARG USERNAME=trist
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN git clone https://github.com/openai/triton.git /home/$USERNAME/triton \
    && git clone https://github.com/tristanheywood/jax-triton /home/$USERNAME/jax-triton \
    && git clone https://github.com/tristanheywood/nimbleGPT /home/$USERNAME/nimbleGPT \
    && git clone https://github.com/tristanheywood/ai-notebooks.git /home/$USERNAME/notebooks

# Not working since switch to pytorch nightly image.
# Install Triton
WORKDIR /home/$USERNAME/triton/python
RUN pip install cmake && PATH="/home/$USERNAME/.local/bin/:$PATH" pip install -e .

# Needed for byobu.
RUN sudo chown -R $USERNAME /home/$USERNAME/.byobu

RUN cd /home/$USERNAME/jax-triton \
    && git remote add upstream https://github.com/jax-ml/jax-triton.git \
    && git fetch upstream \
    && git checkout upstream/mlir


# Install jax-triton (not working)
WORKDIR /home/$USERNAME/jax-triton
RUN pip install -e .

# Install nimbleGPT and deps.
RUN cd /home/$USERNAME/nimbleGPT && pip install -r requirements.txt && pip install -e .

# Pre-install vscode server and some extensions.
# install VS Code (code-server)
RUN sudo apt-get -y install curl && curl -fsSL https://code-server.dev/install.sh | sh

# install VS Code extensions. Python, Copilot, and Jupyter.
RUN code-server --install-extension ms-python.python \
    --install-extension ms-toolsai.jupyter \
    --install-extension github.copilot \
    --install-extension ban.spellright \
    --install-extension sleistner.vscode-fileutils \
