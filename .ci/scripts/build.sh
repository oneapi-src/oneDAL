#! /bin/bash
#===============================================================================
# Copyright 2019 Intel Corporation
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

set -eo pipefail

SCRIPT_DIR=$(dirname $(readlink -f "${BASH_SOURCE[0]}"))
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../../")

show_help() {
    echo "Usage: $0 [--help]"
    echo -e "  --help \t\tShow this help message"
    echo -e "  --compiler \t\tThe compiler toolchain to use. This is a value that is recognised by the oneDAL top level Makefile"
    echo -e "  --optimizations \t\tThe microarchitecture to optimize the build for. This is a value that is recognised by the oneDAL top level Makefile"
    echo -e "  --target \t\tThe oneDAL target to build. This is passed directly to the oneDAL top level Makefile. Multiple targets can be passed by supplying a space-separated string as an argument"
    echo -e "  --backend-config \t\tThe optimised backend CPU library to use. Must be one of [mkl, ref]"
    echo -e "  --conda-env \t\tThe name of the conda environment to load"
    echo -e "  --cross-compile \t\tIndicates that the target platform to build for is not the host platform"
    echo -e "  --plat \t\tThe platform to build for. This is passed to the oneDAL top level Makefile"
}

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --compiler)
        compiler="$2"
        shift;;
        --optimizations)
        optimizations="$2"
        shift;;
        --target)
        target="$2"
        shift;;
        --backend-config)
        backend_config="$2"
        shift;;
        --conda-env)
        conda_env="$2"
        shift;;
        --cross-compile)
        cross_compile="yes"
        ;;
        --plat)
        PLAT="$2"
        shift;;
        --help)
        show_help
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

PLAT=${PLAT:-$(bash ${ONEDAL_DIR}/dev/make/identify_os.sh)}
OS=${PLAT::3}
ARCH=${PLAT:3:3}

backend_config=${backend_config:-mkl}

if [ "${OS}" == "lnx" ]; then
    if [ "${conda_env}" != "" ]; then
        conda_init_path=/usr/share/miniconda/etc/profile.d/conda.sh
        if [ -f ${conda_init_path} ] ; then
            source ${conda_init_path}
            conda activate ${conda_env}
            echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
        fi
    fi
    compiler=${compiler:-gnu}

    #gpu support is only for Linux 64 bit
    if [ "${ARCH}" == "32e" ]; then
            with_gpu="true"
    else
            with_gpu="false"
    fi
elif [ "${OS}" == "mac" ]; then
    if [ "${conda_env}" != "" ]; then
        conda_init_path=/usr/local/miniconda/etc/profile.d/conda.sh
        if [ -f ${conda_init_path} ]; then
            source ${conda_init_path}
            conda activate ${conda_env}
            echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
        fi
    fi
    compiler=${compiler:-clang}
    with_gpu="false"
else
    echo "Error not supported OS: ${OS}"
    exit 1
fi

#setting build parrlelization based on number of thereads
if [ "$(uname)" == "Linux" ]; then
    make_op="-j$(grep -c processor /proc/cpuinfo)"
else
    make_op="-j$(sysctl -n hw.physicalcpu)"
fi

#main actions
echo "Call env scripts"
if [ "${backend_config}" == "mkl" ]; then
    echo "Sourcing MKL env"
    ${ONEDAL_DIR}/dev/download_micromkl.sh with_gpu=${with_gpu}
elif [ "${backend_config}" == "ref" ]; then
    echo "Sourcing ref(openblas) env"
    if [ ! -d "__deps/open_blas" ]; then
        if [ "${optimizations}" == "sve" ] && [ "${cross_compile}" == "yes" ]; then
            ${ONEDAL_DIR}/.ci/env/openblas.sh --target ARMV8 --host_compiler gcc --compiler aarch64-linux-gnu-gcc --cflags -march=armv8-a+sve --cross_compile
        else
            ${ONEDAL_DIR}/.ci/env/openblas.sh
        fi
    fi
else
    echo "Not supported backend env"
fi

# TBB setup
if [[ "${ARCH}" == "32e" ]]; then
    ${ONEDAL_DIR}/dev/download_tbb.sh
elif [[ "${ARCH}" == "arm" ]]; then
    if [[ "${cross_compile}" == "yes" ]]; then
        ${ONEDAL_DIR}/.ci/env/tbb.sh --cross_compile --toolchain_file $(pwd)/.ci/env/arm-gcc-crosscompile-toolchain.cmake --target_arch aarch64
    else
        ${ONEDAL_DIR}/.ci/env/tbb.sh
    fi
fi

if [ "${optimizations}" == "sve" ] && [ "${cross_compile}" == "yes" ]; then
    export CXX=aarch64-linux-gnu-g++
    export CC=aarch64-linux-gnu-gcc 
fi

echo "Calling make"
echo $CXX
echo $CC
echo make ${target:-onedal_c} ${make_op} \
    COMPILER=${compiler} \
    REQCPU="${optimizations}" \
    BACKEND_CONFIG="${backend_config}" \
    PLAT=${PLAT}
make ${target:-onedal_c} ${make_op} \
    COMPILER=${compiler} \
    REQCPU="${optimizations}" \
    BACKEND_CONFIG="${backend_config}" \
    PLAT=${PLAT}
