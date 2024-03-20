#!/bin/bash
#===============================================================================
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

# Function to display help
show_help() {
    echo "Usage: $0 [--help]"
    echo -e "  --help  \t\t\tDisplay this information"
    echo -e "  --CC <bin path> \t\tPass full path to c compiler. Default is GNU gcc."
    echo -e "  --CXX <bin path> \t\tPass full path to c++ compiler. Default is GNU g++."
    echo -e "  --cross-compile <flag> \tPass this flag if cross-compiling."
    echo -e "  --toolchain_file <file> \tPass path to cmake toolchain file. Default is './.ci/env/arm-toolchain.cmake'. Use only with '--cross_compile'"
    echo -e "  --target_arch <name> \t\tTarget architecture name (i.e aarch64, x86_64, etc.) for cross-compilation. Use only with '--cross_compile'"
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --CXX)
        CXX="$2"
        shift;;
        --CC)
        CC="$2"
        shift;;
        --toolchain_file)
        toolchain_file="$2"
        shift;;
        --target_arch)
        target_arch="$2"
        shift;;
        --help)
        show_help
        exit 0
        ;;
        --cross_compile)
        cross_compile="yes"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

set_arch_dir() {
    local arch="$1"
    local arch_dir=""
    if [ "$arch" == "x86_64" ]; then
        arch_dir="intel64"
    elif [ "$arch" == "aarch64" ]; then
        arch_dir="arm"
    else
        echo "Unsupported arch, quitting tbb build script."
        exit 1
    fi
    echo "${arch_dir}"
}

if [ "${cross_compile}" == "yes" ]; then
    toolchain_file=${toolchain_file:-$(pwd)/.ci/env/arm-toolchain.cmake}
    if [ ! -f ${toolchain_file} ]; then
        echo "'${toolchain_file}' file does not exists. Please specify using '--toolchain_file=<path>' argument."
        exit 1
    fi
    target_arch=${target_arch:-aarch64}
    arch_dir=${arch_dir:-$(set_arch_dir "${target_arch}")}
else
    arch=${target_arch:-$(uname -m)}
    if [ "${arch}" == "x86_64" ]; then
        arch_dir="intel64"
    elif [ "${arch}" == "aarch64" ]; then
        arch_dir="arm"
    else
        arch_dir=${arch}
    fi
fi

TBB_VERSION="v2021.10.0"

sudo apt-get update
sudo apt-get install build-essential gcc gfortran cmake -y
git clone --depth 1 --branch ${TBB_VERSION} https://github.com/oneapi-src/oneTBB.git onetbb-src

CoreCount=$(lscpu -p | grep -Ev '^#' | wc -l)

rm -rf __deps/tbb
pushd onetbb-src
rm -rf build
mkdir build
pushd build
if [ "${cross_compile}" == "yes" ]; then
    unset CXX
    unset CC
    echo cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_file} -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb ..
    cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_file} -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb ..
else
    # Set default values for CXX and CC
    CXX="${CXX:-g++}"
    CC="${CC:-gcc}"

    echo "CXX is set to: $CXX"
    echo "CC is set to: $CC"
    cmake -DCMAKE_CXX_COMPILER=${CXX} -DCMAKE_CC_COMPILER=${CC} -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT_PROTOTYPES=OFF -DCMAKE_INSTALL_PREFIX=../../__deps/tbb .. 
fi
make -j${CoreCount}
make install
popd
popd
rm -rf onetbb-src

pushd __deps/tbb
    mkdir -p lnx
    mv lib/ lnx/
    mv include/ lnx/ 
    pushd lnx
        mkdir -p lib/${arch_dir}/gcc4.8
        mv lib/libtbb* lib/${arch_dir}/gcc4.8
    popd
popd 
