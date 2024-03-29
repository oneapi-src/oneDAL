#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
# Copyright contributors to the oneDAL project
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

component=$1

function update {
    sudo apt-get update
}

function add_repo {
    wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
    echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
    sudo add-apt-repository -y "deb https://apt.repos.intel.com/oneapi all main"
    sudo apt-get update
}

function install_dpcpp {
    sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2024.0
    sudo bash -c 'echo libintelocl.so > /etc/OpenCL/vendors/intel-cpu.icd'
}

function install_mkl {
    sudo apt-get install intel-oneapi-mkl-devel
}

function install_clang-format {
    sudo apt-get install -y clang-format-14
    sudo update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-14 100
    sudo update-alternatives --set clang-format /usr/bin/clang-format-14
}

function install_dev-base {
    sudo apt-get install -y gcc-multilib g++-multilib dos2unix tree
}

function install_dev-base-conda {
    conda env create -f .ci/env/environment.yml
}

function install_arm-cross-compilers {
    sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu gfortran-aarch64-linux-gnu
}

function install_arm-emulator {
    sudo apt-get install -y libc6-dev-arm64-cross
    sudo apt-get install -y qemu qemu-user-static
}

if [ "${component}" == "dpcpp" ]; then
    add_repo
    install_dpcpp
elif [ "${component}" == "mkl" ]; then
    add_repo
    install_mkl
elif [ "${component}" == "arm-compiler" ]; then
    update
    install_arm-cross-compilers
elif [ "${component}" == "arm-emulator" ]; then
    update
    install_arm-emulator
elif [ "${component}" == "clang-format" ]; then
    update
    install_clang-format
elif [ "${component}" == "dev-base" ]; then
    update
    install_dev-base
    install_dev-base-conda
else
    echo "Usage:"
    echo "   $0 [dpcpp|mkl|arm-compiler|clang-format|dev-base]"
    exit 1
fi
