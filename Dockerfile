# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev


# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Additionally install the jupyterlab extension

RUN --mount=type=cache,target=/root/.cache/uv \
    uv add jupyterlab ipywidgets

# Verify Jupyter is installed and in PATH
RUN which jupyter || echo "Jupyter not found in PATH"

    # Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []
EXPOSE 8888

# Enter the examples directory
# Default command to start JupyterLab (using JSON array format)
CMD ["jupyter","lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

# How to build and run the image
# docker build -t skfolio-jupyterlab .
# docker run -p 8888:8888 -v <path-to-your-folder-containing-data>:/app/data -it skfolio