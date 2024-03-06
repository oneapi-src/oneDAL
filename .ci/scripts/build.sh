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

# Obtain platform, OS and arch details automatically
PLATFORM=$(bash dev/make/identify_os.sh)
OS=${PLATFORM::3}
ARCH=${PLATFORM:3:3}

# set default values for optimisation based on arch, these values can be overidden by passed arguments to the script.
if [[ "${ARCH}" == "32e" ]]
then
optimizations=${optimizations:-avx2}
elif [[ "${ARCH}" == "arm" ]]
then
optimizations=${optimizations:-sve}
else
echo "Unknown architecture '${ARCH}'"
exit 1
fi

# set PLAT based on OS
if [[ "${OS}" == "lnx" ]]; then
    if [[ "${ARCH}" == "32e" ]]; then
        PLAT=lnx32e
    elif [[ "${ARCH}" == "arm" ]]; then
        PLAT=lnxarm
    fi
elif [[ "${OS}" == "win" ]]; then
    if [[ "${ARCH}" == "32e" ]]; then
        PLAT=win2e
    fi
elif [[ "${OS}" == "mac" ]]; then
    if [[ "${ARCH}" == "32e" ]]; then
        PLAT=mac32e
    fi
fi

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --compiler)
        compiler="$2"
        ;;
        --optimizations)
        optimizations="$2"
        ;;
        --target)
        target="$2"
        ;;
        --backend_config)
        backend_config="$2"
        ;;
        --conda-env)
        conda_env="$2"
        ;;
        --cross_compile)
        cross_compile="$2"
        ;;
        --arch)
        ARCH="$2"
        ;;
        --plat)
        PLAT="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

backend_config=${backend_config:-mkl}
GLOBAL_RETURN=0

if [ "${OS}" == "lnx" ]; then
    source /usr/share/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        conda activate ${conda_env}
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-gnu}

    #gpu support is only for Linux 64 bit
    if [ "${ARCH}" == "32e" ]; then
            with_gpu="true"
    else
            with_gpu="false"
    fi
elif [ "${OS}" == "mac" ]; then
    source /usr/local/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        conda activate ${conda_env}
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
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
    $(pwd)/dev/download_micromkl.sh with_gpu=${with_gpu}
elif [ "${backend_config}" == "ref" ]; then
    echo "Sourcing ref(openblas) env"
    if [ ! -d "__deps/open_blas" ]; then
        if [ "${optimizations}" == "sve" ] && [ "${cross_compile}" == "yes" ]; then
            $(pwd)/.ci/env/openblas.sh --target ARMV8 --host_compiler gcc --compiler aarch64-linux-gnu-gcc --cflags -march=armv8-a+sve --cross_compile yes
        else
            $(pwd)/.ci/env/openblas.sh
        fi
    fi
else
    echo "Not supported backend env"
fi

TBB setup
if [[ "${ARCH}" == "32e" ]]; then
    $(pwd)/dev/download_tbb.sh
elif [[ "${ARCH}" == "arm" ]]; then
    if [[ "${cross_compile}" == "yes" ]]; then
        $(pwd)/.ci/env/tbb.sh --toolchain_file $(pwd)/.ci/env/arm-toolchain.cmake --arch_dir arm --cross_compile yes
    else
        $(pwd)/.ci/env/tbb.sh
    fi
fi

if [ "${optimizations}" == "sve" ] && [ "${cross_compile}" == "yes" ]; then
    export CXX=aarch64-linux-gnu-g++
    export CC=aarch64-linux-gnu-gcc 
fi

echo "Calling make"
echo $CXX
echo $CC
echo make ${target:-daal_c} ${make_op} \
    COMPILER=${compiler} \
    REQCPU="${optimizations}" \
    BACKEND_CONFIG="${backend_config}" \
    PLAT=$PLAT
make ${target:-daal_c} ${make_op} \
    COMPILER=${compiler} \
    REQCPU="${optimizations}" \
    BACKEND_CONFIG="${backend_config}" \
    PLAT=${PLAT}
err=$?

if [ ${err} -ne 0 ]; then
    status_ex="$(date +'%H:%M:%S') BUILD FAILED with errno ${err}"
    GLOBAL_RETURN=${err}
fi

exit ${GLOBAL_RETURN}
