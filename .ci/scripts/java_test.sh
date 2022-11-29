#! /bin/bash
#===============================================================================
# Copyright 2022 Intel Corporation
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
        --platform)
        platform="$2"
        ;;
        --release-dir)
        release_dir="$2"
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

OS=${platform::3}
ARCH=${platform:3:3}

if [ "${OS}" == "lnx" ]; then
    source /usr/share/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        echo "conda activate ${conda_env}"
        conda activate ${conda_env}
    fi
    export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
    java_os_name="linux"
elif [ "${OS}" == "mac" ]; then
    source /usr/local/miniconda/etc/profile.d/conda.sh
    if [ "${conda_env}" != "" ]; then
        echo "conda activate ${conda_env}"
        conda activate ${conda_env}
    fi
    java_os_name="darwin"
else
    echo "Error not supported OS: ${OS}"
    exit 1
fi

echo "Set Java PATH and CPATH from JAVA_HOME=${JAVA_HOME}"
export PATH=$JAVA_HOME/bin:$PATH
export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/${java_os_name}:$CPATH

TESTING_RETURN=0
if [ "${ARCH}" == "32" ]; then
    full_arch=ia32
else
    full_arch=intel64
fi

#setup env for DAL
source ${release_dir}/daal/latest/env/vars.sh

#setup env for TBB
export TBBROOT=$(pwd)/__deps/tbb/${OS}
export CPATH=${TBBROOT}/include:$CPATH
export LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${TBBROOT}/lib:${LIBRARY_PATH}
if [ "${OS}" == "mac" ]; then
    export DYLD_LIBRARY_PATH=${TBBROOT}/lib:${DYLD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${TBBROOT}/lib/${full_arch}/gcc4.8:${LD_LIBRARY_PATH}
fi

cd ${release_dir}/daal/latest/examples/daal/java
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
