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

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --test-kind)
        TEST_KIND="$2"
        shift;;
        --build-dir)
        BUILD_DIR="$2"
        shift;;
        --compiler)
        compiler="$2"
        shift;;
        --interface)
        interface="$2"
        shift;;
        --conda-env)
        conda_env="$2"
        shift;;
        --build_system)
        build_system="$2"
        shift;;
        --backend)
        backend="$2"
        shift;;
        --cross_compile)
        cross_compile="yes"
        ;;
        --arch)
        ARCH="$2"
        shift;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
done

#Global exit code for testing script
TESTING_RETURN=0

if [ "$ARCH" == "32e" ]; then
    full_arch=intel64
    arch_dir=intel_intel64
elif [ "$ARCH" == "arm" ]; then
    full_arch=arm
    arch_dir=arm_aarch64
else
    echo "Unknown architecture ${ARCH} detected for platform ${PLATFORM}"
    exit 1
fi

build_system=${build_system:-cmake}
backend=${backend:-mkl}

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
if [ "${ARCH}" == "arm" ]; then
    export ARCH_ONEDAL=aarch64
fi
source ${BUILD_DIR}/daal/latest/env/vars.sh

#setup env for TBB
# Check if path exists
tbb_root_path=$(pwd)/__deps/tbb/${OS}
if [[ -e $tbb_root_path ]]; then
    export TBBROOT=$tbb_root_path
else
    export TBBROOT=$(pwd)/__deps/tbb
fi

# Check if path exists
tbb_dir_path=${TBBROOT}/lib/cmake/tbb
if [[ -e $tbb_dir_path ]]; then
    export TBB_DIR=$tbb_dir_path
else
    export TBB_DIR=${TBBROOT}/lib/cmake/TBB
fi
export CPATH=${TBBROOT}/include:$CPATH
export CMAKE_MODULE_PATH=${TBB_DIR}:${CMAKE_MODULE_PATH}
export TOOLCHAIN_FOLDER=$(pwd)/.ci/env

if [ "${OS}" == "mac" ]; then
    export DYLD_LIBRARY_PATH=${TBBROOT}/lib:${DYLD_LIBRARY_PATH}
    export LIBRARY_PATH=${TBBROOT}/lib:${LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${TBBROOT}/lib:${LD_LIBRARY_PATH}
    export LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${TBBROOT}/lib:${LIBRARY_PATH}
fi

interface=${interface:-daal/cpp}
cd "${BUILD_DIR}/daal/latest/${TEST_KIND}/${interface}"

for link_mode in ${link_modes}; do
    if [ "${link_mode}" == "static" ]; then
        lib_ext="a"
        l="lib"
    elif [ "${link_mode}" == "dynamic" ]; then
        lib_ext="so"
        if [ "${OS}" == "lnx" ]; then
            l="so"
        else
            l="dylib"
        fi
    fi
    if [ "$build_system" == "cmake" ]; then
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

        if [ "${cross_compile}" == "yes" ]; then
            unset CXX
            unset CC
            echo "============== Configuration: =============="
            echo Compiler:  ${compiler}
            echo Link mode: ${link_mode}
            echo Using Cross-compiler.
            echo "============================================"
        else
            echo "============== Configuration: =============="
            echo Compiler:  ${compiler}
            echo Link mode: ${link_mode}
            echo CC: ${CC}
            echo CXX: ${CXX}
            echo "============================================"
        fi

        if [ -d "Build" ]; then
            rm -rf Build/*
        else
            mkdir Build
        fi

        ref_backend="OFF"
        if [ "${backend}" == "ref" ]; then
            ref_backend="ON"
        fi

        if [ "${cross_compile}" == "yes" ]; then
            if [ "${ARCH}" == "arm" ]; then
                cmake -B Build -S . -G "Unix Makefiles" -DCMAKE_HOST_SYSTEM_PROCESSOR=aarch64 -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FOLDER}/arm-gcc-crosscompile-toolchain.cmake -DCMAKE_PREFIX_PATH=${TBBROOT} -DONEDAL_LINK=${link_mode} -DTBB_DIR=${TBB_DIR} -DREF_BACKEND=${ref_backend} -DOUTPUT_DIR=arm
            fi
        else
            cmake -B Build -S . -G "Unix Makefiles" -DONEDAL_LINK=${link_mode} -DTBB_DIR=${TBB_DIR} -DREF_BACKEND=${ref_backend}
        fi

        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') CMAKE GENERATE FAILED\t\t"
            TESTING_RETURN=${err}
            continue
        fi
        make ${make_op} -C Build
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t"
            TESTING_RETURN=${err}
            continue
        fi
        output_result=
        err=
        cmake_results_dir="_cmake_results/${arch_dir}_${lib_ext}"
        for p in ${cmake_results_dir}/*; do
            e=$(basename "$p")
            if [ "${ARCH}" == "arm" ] && [ "${cross_compile}" == "yes" ]; then
                qemu-aarch64-static -L /usr/aarch64-linux-gnu/ ${p} 2>&1 > ${e}.res
            else
                ${p} 2>&1 > ${e}.res
            fi
            err=$?
            output_result=$(cat ${e}.res)
            mv -f ${e}.res ${cmake_results_dir}/
            status_ex=
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
    else
        build_command="make ${make_op} ${l}${full_arch} mode=build compiler=${compiler}"
        echo "Building ${TEST_KIND} ${build_command}"
        (${build_command})
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t${link_mode}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') BUILD COMPLETED\t\t${link_mode}"
        fi
        run_command="make ${l}${full_arch} mode=run compiler=${compiler}"
        echo "Running ${TEST_KIND} ${run_command}"
        (${run_command})
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') RUN FAILED\t\t${link_mode} with errno ${err}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') RUN PASSED\t\t${link_mode}"
        fi
    fi
done

#exit with overall testing status
exit ${TESTING_RETURN}
