#!/bin/bash
set -ex

DOCKER_IMAGE=rocm/pytorch

docker run --rm \
   -v $(pwd):/sgl-kernel \
   ${DOCKER_IMAGE} \
   bash -c "
   # Install CMake (version >= 3.26) - Robust Installation
   export CMAKE_VERSION_MAJOR=3.31
   export CMAKE_VERSION_MINOR=1
   echo \"Downloading CMake from: https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-${ARCH}.tar.gz\"
   wget https://cmake.org/files/v\${CMAKE_VERSION_MAJOR}/cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
   tar -xzf cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64.tar.gz
   mv cmake-\${CMAKE_VERSION_MAJOR}.\${CMAKE_VERSION_MINOR}-linux-x86_64 /opt/cmake
   export PATH=/opt/cmake/bin:\$PATH

   # Debugging CMake
   echo \"PATH: \$PATH\"
   which cmake
   cmake --version

   /opt/venv/bin/pip install --no-cache-dir ninja setuptools==75.0.0 wheel==0.41.0 numpy uv scikit-build-core && \

   cd /sgl-kernel && \
   /opt/venv/bin/python rocm_hipify.py && \
   /opt/venv/bin/python -m uv build --wheel -Cbuild-dir=build . --color=always --no-build-isolation && \
   ./rename_wheels_rocm.sh
"
