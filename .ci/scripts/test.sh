#! /bin/bash
#===============================================================================
# Copyright 2019-2021 Intel Corporation
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
ALGORITHM="kmeans_dense_batch"
if [ "${ARCH}" == "32" ]; then
    full_arch=ia32
else
    full_arch=intel64
fi

if [ "${OS}" == "lnx" ]; then
    compiler=${compiler:-gnu}
elif [ "${OS}" == "mac" ]; then
    compiler=${compiler:-clang}

else
    echo "Error not supported OS: ${OS}"
    exit 1
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

cd "${BUILD_DIR}/daal/latest/examples/daal/cpp"

for threading in parallel sequential; do
    for link_mode in static dynamic; do
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
        build_command="make ${l}${full_arch} example=${ALGORITHM} mode=build  compiler=${compiler} threading=${threading}"
        echo "Building example ${build_command}"
        (${build_command} >> ${ALGORITHM}.log)
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') BUILD FAILED\t\t${ALGORITHM}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') BUILD COMPLETED\t\t${ALGORITHM}"
        fi
        run_command="make ${l}${full_arch} example=${ALGORITHM} mode=run compiler=${compiler} threading=${threading}"
        echo "Running example ${run_command}"
        (${run_command} >> ${ALGORITHM}.log)
        err=$?
        if [ ${err} -ne 0 ]; then
            echo -e "$(date +'%H:%M:%S') FAILED\t\t${ALGORITHM} with errno ${err}"
            TESTING_RETURN=${err}
            continue
        else
            echo -e "$(date +'%H:%M:%S') PASSED\t\t${ALGORITHM}"
        fi
    done
done

#exit with overall testing status
exit ${TESTING_RETURN}
