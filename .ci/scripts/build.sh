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

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
ONEDAL_DIR=$(readlink -f "${SCRIPT_DIR}/../../")

show_help() {
    echo "Usage: $0 [--help]"
    column -t -s":" <<< '--help:Show this help message
--compiler:The compiler toolchain to use. This is a value that is recognised by the oneDAL top level Makefile, and must be one of [gnu, clang, icx]
--optimizations:The microarchitecture to optimize the build for. This is a value that is recognised by the oneDAL top level Makefile
--target:The oneDAL target to build. This is passed directly to the oneDAL top level Makefile. Multiple targets can be passed by supplying a space-separated string as an argument
--backend-config:The optimised backend CPU library to use. Must be one of [mkl, ref]
--conda-env:The name of the conda environment to load
--cross-compile:Indicates that the target platform to build for is not the host platform
--plat:The platform to build for. This is passed to the oneDAL top level Makefile
'
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

PLAT=${PLAT:-$(bash "${ONEDAL_DIR}"/dev/make/identify_os.sh)}
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

#setting build parallelization based on number of threads
if [ "$(uname)" == "Linux" ]; then
    make_op="-j$(nproc --all)"
else
    make_op="-j$(sysctl -n hw.physicalcpu)"
fi

# Override the compilers. We know which compilers we want in
# the case that we are using a GNU or LLVM toolchain
if [ "${ARCH}" == "arm" ] && [ "${cross_compile}" == "yes" ] && [ "${compiler}" == "gnu" ] ; then
    export CXX=aarch64-linux-gnu-g++
    export CC=aarch64-linux-gnu-gcc
elif [ "${compiler}" == "clang" ] ; then
    export CXX=clang++
    export CC=clang
elif [ "${compiler}" == "gnu" ] ; then
    export CXX=g++
    export CC=gcc
elif [ "${compiler}" == "icx" ] ; then
    export CXX=icpx
    export CC=icx
else
    echo "Unsupported compiler '${compiler}'"
    exit 1
fi

#main actions
echo "Call env scripts"
if [ "${backend_config}" == "mkl" ]; then
    echo "Sourcing MKL env"
    "${ONEDAL_DIR}"/dev/download_micromkl.sh with_gpu="${with_gpu}"
elif [ "${backend_config}" == "ref" ]; then
    echo "Sourcing ref(openblas) env"
    if [ ! -d "${ONEDAL_DIR}/__deps/openblas_${ARCH}" ]; then
        if [ "${optimizations}" == "sve" ] && [ "${cross_compile}" == "yes" ]; then
            openblas_options=(--target ARMV8
                --host-compiler gcc
                --compiler "${CC}"
                --cflags -march=armv8-a+sve
                --cross-compile
                --target-arch "${ARCH}")
            echo "${ONEDAL_DIR}"/.ci/env/openblas.sh "${openblas_options[@]}"
            "${ONEDAL_DIR}"/.ci/env/openblas.sh "${openblas_options[@]}"
        else
            "${ONEDAL_DIR}"/.ci/env/openblas.sh --target-arch "${ARCH}"
        fi
    fi
    export OPENBLASROOT="${ONEDAL_DIR}/__deps/openblas_${ARCH}"
else
    echo "Not supported backend env"
fi

# TBB setup
if [[ "${ARCH}" == "32e" ]]; then
    "${ONEDAL_DIR}"/dev/download_tbb.sh
elif [[ "${ARCH}" == "arm" ]]; then
    if [[ "${cross_compile}" == "yes" ]]; then
        tbb_options=(--cross-compile
          --toolchain-file
          "${ONEDAL_DIR}"/.ci/env/arm-gcc-crosscompile-toolchain.cmake
          --target-arch aarch64
        )
        echo "${ONEDAL_DIR}"/.ci/env/tbb.sh "${tbb_options[@]}"
        "${ONEDAL_DIR}"/.ci/env/tbb.sh "${tbb_options[@]}"
    else
        "${ONEDAL_DIR}"/.ci/env/tbb.sh
    fi
    export TBBROOT="$ONEDAL_DIR/__deps/tbb-aarch64"
    export LD_LIBRARY_PATH=${TBBROOT}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

make_options=("${target:-onedal_c}"
    "${make_op}"
    COMPILER="${compiler}"
    REQCPU="${optimizations}"
    BACKEND_CONFIG="${backend_config}"
    PLAT="${PLAT}"
)

echo "Calling make"
echo "CXX=$CXX"
echo "CC=$CC"
echo make "${make_options[@]}"
make "${make_options[@]}"
