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
        --compiler)
        compiler="$2"
        ;;
        --optimizations)
        optimizations="$2"
        ;;
        --target)
        target="$2"
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

PLATFORM=$(bash dev/make/identify_os.sh)
OS=${PLATFORM::3}
ARCH=${PLATFORM:3:3}

optimizations=${optimizations:-avx2}
GLOBAL_RETURN=0

if [ "${OS}" == "lnx" ]; then
    source /usr/share/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        conda activate ${conda_env}
        echo "conda '${conda_env}' env activated at ${CONDA_PREFIX}"
    fi
    compiler=${compiler:-gnu}
    java_os_name="linux"
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
    java_os_name="darwin"
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
echo "Call mkl and tbb scripts"
$(pwd)/dev/download_micromkl.sh with_gpu=${with_gpu}
$(pwd)/dev/download_tbb.sh
echo "Set Java PATH and CPATH from JAVA_HOME=${JAVA_HOME}"
export PATH=$JAVA_HOME/bin:$PATH
export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/${java_os_name}:$CPATH
echo "Calling make"
make ${target:-daal_c} ${make_op} \
    COMPILER=${compiler} \
    REQCPU="${optimizations}"
err=$?

if [ ${err} -ne 0 ]; then
    status_ex="$(date +'%H:%M:%S') BUILD FAILED with errno ${err}"
    GLOBAL_RETURN=${err}
fi

exit ${GLOBAL_RETURN}
