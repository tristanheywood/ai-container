# Use the official Python base image
FROM python:3.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        vim \
        git \
        htop \
        curl \
        sudo

# Install common Python data science packages
RUN pip install --upgrade pip && \
    pip install \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn \
        jupyterlab \
        ipython

# Create a non-root user with sudo privileges
RUN useradd -m -s /bin/bash trist && \
    echo "trist ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to the non-root user
USER trist

# Set environment variables for the non-root user
ENV HOME="/home/trist"
ENV POETRY_HOME="${HOME}/.poetry"
ENV PATH="${PATH}:${POETRY_HOME}/bin"

# Install Poetry as the non-root user
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set the working directory
WORKDIR /home/trist

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
