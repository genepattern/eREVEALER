# Base image with specific version
FROM debian:bullseye-slim

# Set environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHON_VERSION=3.9.13 \
    PYTHON_PIP_VERSION=22.3.1 \
    PYTHON_SETUPTOOLS_VERSION=65.5.1 \
    PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py \
    PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6

# Install dependencies
RUN set -ex; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        gnupg \
        dirmngr \
        xz-utils \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev; \
    rm -rf /var/lib/apt/lists/*

# Install Python
RUN set -ex; \
    wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
    mkdir -p /usr/src/python; \
    tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
    rm python.tar.xz; \
    cd /usr/src/python; \
    gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
    ./configure \
        --build="$gnuArch" \
        --enable-loadable-sqlite-extensions \
        --enable-optimizations \
        --enable-option-checking=fatal \
        --enable-shared \
        --with-system-expat \
        --without-ensurepip; \
    make -j "$(nproc)"; \
    make install; \
    cd /; \
    rm -rf /usr/src/python; \
    ldconfig; \
    python3 --version

# Symlink Python executables
RUN set -ex; \
    for src in idle3 pydoc3 python3 python3-config; do \
        dst="$(echo "$src" | tr -d 3)"; \
        [ -s "/usr/local/bin/$src" ]; \
        [ ! -e "/usr/local/bin/$dst" ]; \
        ln -svT "$src" "/usr/local/bin/$dst"; \
    done

# Install pip and setuptools
RUN set -ex; \
    wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
    echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum -c -; \
    python3 get-pip.py \
        --disable-pip-version-check \
        --no-cache-dir \
        --no-compile \
        "pip==$PYTHON_PIP_VERSION" \
        "setuptools==$PYTHON_SETUPTOOLS_VERSION"; \
    rm -f get-pip.py; \
    pip --version

# Set work directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install -r /app/requirements.txt

# Install REVEALER (if it's available via pip)
RUN pip install REVEALER

# Set the default command
CMD ["python3"]

# Maintainer label
LABEL maintainer=jim095@ucsd.edu