#!/usr/bin/env bash

set -e

pip3 install --no-cache --upgrade pip

# ========================================================= #
# 0. Checking Latest Available Versions
# ========================================================= #

## 1.1 JAX
pip3 index versions --pre jax \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html

## 1.2 Triton-Nightly
pip3 index versions --pre triton-nightly \
    --index-url "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/"

# ========================================================= #
# 1. Remove all potential previous installs
# ========================================================= #

pip3 uninstall -y triton triton-nightly jax jaxlib jax-triton

# ========================================================= #
# 2. Installing OpenAI Triton and Dependencies
# ========================================================= #

# Dependency for triton
pip3 install --no-cache torch --index-url https://download.pytorch.org/whl/cu121
pip3 uninstall -y triton triton-nightly

# OpenAI Triton
TRITON_VERSION="2.1.0.post20231217061226"

pip3 install --no-cache --pre --upgrade \
    --extra-index-url "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/" \
    triton-nightly==${TRITON_VERSION}

# Triton JAX
JAX_TRITON_COMMIT="4a5791d00809ac96f92d42a80a4f52501d1421be"

pip3 install --no-cache --no-deps "jax-triton @ git+https://github.com/jax-ml/jax-triton.git@${JAX_TRITON_COMMIT}"
pip3 install --no-cache absl-py

# ========================================================= #
# 3. Installing OpenAI Triton and Dependencies
# ========================================================= #

JAX_VERSION="0.4.24.dev20240109"

# JAX NIGHTLY
# https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
pip3 install --no-cache -U --pre jax[cuda12_local]==${JAX_VERSION} \
    -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html \
    -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
