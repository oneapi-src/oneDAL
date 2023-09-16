#===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

FROM ubuntu:22.04

ARG workdirectory="/sources/oneDAL"

ADD ../../ ${workdirectory}

WORKDIR ${workdirectory}

#Env setup
RUN apt-get update && \
      apt-get -y install sudo wget gnupg git make python3-setuptools doxygen

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Installing environment for base development dependencies
RUN .ci/env/apt.sh dev-base

# Installing environment for DPCPP development dependencies
RUN .ci/env/apt.sh dpcpp

# Installing environment for clang-format
RUN .ci/env/apt.sh clang-format

# Installing environment for bazel
RUN .ci/env/bazelisk.sh

# Installing openBLAS dependency
RUN .ci/env/openblas.sh

# Installing MKL dependency
RUN ./dev/download_micromkl.sh

# Installing TBB dependency
RUN ./dev/download_tbb.sh

# Installing doc build dependency
#RUN pip3 install -r docs/requirements.txt
