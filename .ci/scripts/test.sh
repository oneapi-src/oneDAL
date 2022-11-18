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
    compiler=${compiler:-gnu}
    link_modes="static dynamic"
elif [ "${OS}" == "mac" ]; then
    compiler=${compiler:-clang}
    if [ "${compiler}" == "gnu" ]; then
        # TODO: fix static linking with gnu on mac
        link_modes="dynamic"
    else
        link_modes="static dynamic"
    fi
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
cd "${BUILD_DIR}/daal/latest/examples/${interface}"

for link_mode in ${link_modes}; do
    # Release Examples testing
    if [ "${link_mode}" == "static" ]; then
        l="lib"
    elif [ "${link_mode}" == "dynamic" ]; then
        if [ "${OS}" == "lnx" ]; then
            l="so"
        else
            l="dylib"
        fi
    fi
    build_command="make ${make_op} ${l}${full_arch} mode=build compiler=${compiler}"
    echo "Building examples ${build_command}"
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
    echo "Running examples ${run_command}"
    (${run_command})
    err=$?
    if [ ${err} -ne 0 ]; then
        echo -e "$(date +'%H:%M:%S') RUN FAILED\t\t${link_mode} with errno ${err}"
        TESTING_RETURN=${err}
        continue
    else
        echo -e "$(date +'%H:%M:%S') RUN PASSED\t\t${link_mode}"
    fi
done

#exit with overall testing status
exit ${TESTING_RETURN}