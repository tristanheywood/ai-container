# Building the container

We split the build up into a cache-able part and a non cacheable part. The non-cacheable
part pulls git repos.

```
docker build base-image -t ai-container-base
docker build dev-image --no-cache -t ai-container
```

Note: `--no-cache` can be omitted if you pull and build the repos inside the container.

Builds an image called 'ai-container-base'.

## In-container setup steps (todo: move these into the docker file)

```
# Make byobu work
sudo chown -R trist /home/trist/.byobu
# Make Nsight work
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
# Set jax-triton upstream
cd ~/jax-triton && git remote add upstream https://github.com/jax-ml/jax-triton.git && git fetch upstream
# Install jax-triton
cd ~/jax-triton && pip install -e .
# Install nimbleGPT and deps
cd ~/nimbleGPT && pip install -r requirements.txt && pip install -e .
# Let conda install packages
sudo chown trist -R /opt/conda

```

# Running the container for the first time

`docker run -it --gpus all --cap-add SYS_ADMIN --privileged ai-container byobu`

# Resuming the container

```
docker start elegant_maxwell
docker attach elegant_maxwell
```

# Check container is set up correctly

## GPU is available

`nvidia-smi`

## Nsight profiling is enabled

`nsys status -e`

Should end with "Sampling Environment: OK"

## Test Triton

```
cd ~/triton
pip install -e '.[tests]'
pytest -vs test/unit/
```

## Test jax-triton

```
cd ~/jax-triton
pip install pytest absl-py
pytest tests/
```