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
    sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2024.1
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

function install_gnu-cross-compilers {
    sudo apt-get install -y "gcc-$1-linux-gnu" "g++-$1-linux-gnu" "gfortran-$1-linux-gnu"
}

function install_qemu_emulation_apt {
    sudo apt-get install -y qemu-user-static
}

function install_qemu_emulation_deb {
    qemu_deb=qemu-user-static_8.2.3+ds-2_amd64.deb
    set -eo pipefail
    wget http://ftp.debian.org/debian/pool/main/q/qemu/${qemu_deb}
    sudo dpkg -i ${qemu_deb}
    sudo systemctl restart systemd-binfmt.service
    set +eo pipefail
}

function install_llvm_version {
    sudo apt-get install -y curl
    curl -o llvm.sh https://apt.llvm.org/llvm.sh
    chmod u+x llvm.sh
    sudo ./llvm.sh "$1"
    sudo update-alternatives --install /usr/bin/clang clang "/usr/bin/clang-$1" "${1}00"
    sudo update-alternatives --install /usr/bin/clang++ clang++ "/usr/bin/clang++-$1" "${1}00"
}

function build_sysroot {
    # Usage:
    #   build_sysroot working-dir arch os-name out-dir
    # where the architecture and OS name need to be recognised by debootstrap,
    # e.g. arch=arm64, os-name=jammy. The output directory path is relative to
    # the working directory
    mkdir -p "$1"
    pushd "$1" || exit
    sudo apt-get install -y debootstrap build-essential
    sudo debootstrap --arch="$2" --verbose --include=fakeroot,symlinks,libatomic1 --resolve-deps --variant=minbase --components=main,universe "$3" "$4"
    sudo chroot "$4" symlinks -cr .
    sudo chown "${USER}" -R "$4"
    rm -rf "${4:?}"/{dev,proc,run,sys,var}
    rm -rf "${4:?}"/usr/{sbin,bin,share}
    rm -rf "${4:?}"/usr/lib/{apt,gcc,udev,systemd}
    rm -rf "${4:?}"/usr/libexec/gcc
    popd || exit
}

if [ "${component}" == "dpcpp" ]; then
    add_repo
    install_dpcpp
elif [ "${component}" == "mkl" ]; then
    add_repo
    install_mkl
elif [ "${component}" == "gnu-cross-compilers" ]; then
    update
    install_gnu-cross-compilers "$2"
elif [ "${component}" == "clang-format" ]; then
    update
    install_clang-format
elif [ "${component}" == "dev-base" ]; then
    update
    install_dev-base
    install_dev-base-conda
elif [ "${component}" == "qemu-apt" ]; then
    update
    install_qemu_emulation_apt
elif [ "${component}" == "qemu-deb" ]; then
    update
    install_qemu_emulation_deb
elif [ "${component}" == "llvm-version" ] ; then
    update
    install_llvm_version "$2"
elif [ "${component}" == "build-sysroot" ] ; then
    update
    build_sysroot "$2" "$3" "$4" "$5"
else
    echo "Usage:"
    echo "   $0 [dpcpp|mkl|gnu-cross-compilers|clang-format|dev-base|qemu-apt|qemu-deb|llvm-version|build-sysroot]"
    exit 1
fi
