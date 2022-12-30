#! /bin/bash
#===============================================================================
# Copyright 2019 Intel Corporation
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

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --test-kind)
        TEST_KIND="$2"
        ;;
        --build-dir)
        BUILD_DIR="$2"
        ;;
        --platform)
        platform="$2"
        ;;
        --compiler)
        compiler="$2"
        ;;
        --interface)
        interface="$2"
        ;;
        --conda-env)
        conda_env="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

#Global exit code for testing script
TESTING_RETURN=0
OS=${platform::3}
ARCH=${platform:3:3}
if [ "${ARCH}" == "32" ]; then
    full_arch=ia32
else
    full_arch=intel64
fi

if [ "${OS}" == "lnx" ]; then
    source /usr/share/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        conda activate ${conda_env}
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-gnu}
    link_modes="static dynamic"
elif [ "${OS}" == "mac" ]; then
    source /usr/local/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        conda activate ${conda_env}
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-clang}
    link_modes="static dynamic"
else
    echo "Error not supported OS: ${OS}"
    exit 1
fi

if [ "$(uname)" == "Linux" ]; then
    make_op="-j$(grep -c processor /proc/cpuinfo)"
else
    make_op="-j$(sysctl -n hw.physicalcpu)"
fi

#setup env for DAL
source ${BUILD_DIR}/daal/latest/env/vars.sh

#setup env for TBB
export TBBROOT=$(pwd)/__deps/tbb/${OS}
export CPATH=${TBBROOT}/include:$CPATH
export LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${TBBROOT}/lib:${LIBRARY_PATH}
if [ "${OS}" == "mac" ]; then
    export DYLD_LIBRARY_PATH=${TBBROOT}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${LD_LIBRARY_PATH}
fi

interface=${interface:-daal/cpp}
cd "${BUILD_DIR}/daal/latest/${TEST_KIND}/${interface}"

if [ "${interface}" == "daal/java" ]; then
    bash launcher.sh
    err=$?
    if [ ${err} -ne 0 ]; then
        echo -e "$(date +'%H:%M:%S') EXAMPLES FAILED\t\t with errno ${err}"
        TESTING_RETURN=${err}
        continue
    else
        echo -e "$(date +'%H:%M:%S') EXAMPLES PASSED\t\t"
    fi

    exit ${TESTING_RETURN}
fi

for link_mode in ${link_modes}; do
    if [ "${link_mode}" == "static" ]; then
        lib_ext="a"
        l="lib"
    elif [ "${link_mode}" == "dynamic" ]; then
        if [ "${OS}" == "lnx" ]; then
            lib_ext="so"
            l="so"
        else
            lib_ext="dylib"
            l="dylib"
        fi
    fi
    if [[ ${compiler} == gnu ]]; then
        export CC=gcc
        export CXX=g++
    elif [[ ${compiler} == clang ]]; then
        export CC=clang
        export CXX=clang++
    elif [[ ${compiler} == icx ]]; then
        export CC=icx
        export CXX=icpx
    fi
    echo "============== Configuration: =============="
    echo Compiler:  ${compiler}
    echo Link mode: ${link_mode}
    echo CMAKE_C_COMPILE: ${CC}
    echo CMAKE_CXX_COMPILER: ${CXX}
    echo "============================================"

    if [ -d "Build" ]; then
        rm -rf ./Build/*
    else
        mkdir Build
    fi

    cmake -B./Build -S. -G "Unix Makefiles" -DTARGET_LINK=${link_mode} -DSET_TBB_MANUALLY=yes
    err=$?
    if [ ${err} -ne 0 ]; then
        echo -e "$(date +'%H:%M:%S') CMAKE GENERATE FAILED\t\t"
        TESTING_RETURN=${err}
        continue
    fi
    make -C Build
    err=$?
    if [ ${err} -ne 0 ]; then
        echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t"
        TESTING_RETURN=${err}
        continue
    fi
    local output_result=
    local err=
    local cmake_results_dir="_cmake_results/intel_intel64_${lib_ext}"
    for p in ${cmake_results_dir}/*; do
        e=$(basename "$p")
        ${p} 2>&1 > ${e}.res
        err=$?
        output_result=$(cat ${e}.res)
        mv -f ${e}.res ${cmake_results_dir}/
        local status_ex=
        if [ ${err} -ne 0 ]; then
            echo "${output_result}"
            status_ex="$(date +'%H:%M:%S') FAILED\t\t${e} with errno ${err}"
            TESTING_RETURN=${err}
            continue
        else
            echo "${output_result}" | grep -i "error\|warn"
            status_ex="$(date +'%H:%M:%S') PASSED\t\t${e}"
        fi
        echo -e $status_ex
    done
done

#exit with overall testing status
exit ${TESTING_RETURN}