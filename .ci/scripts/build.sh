#! /bin/bash

#===============================================================================
# Copyright 2019 - 2020 Intel Corporation
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
        --compiler)
        compiler="$2"
        ;;
        --target)
        target="$2"
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
CPU_OPTIMIZATIONS="avx2"

if [ "${OS}" == "lnx" ]; then
    compiler=${compiler:-gnu}
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    java_os_name="linux"
    #gpu support is only for Linux 64 bit
    if [ "${ARCH}" == "32e" ]; then
            with_gpu="true"
    else
            with_gpu="false"
    fi
elif [ "${OS}" == "mac" ]; then
    compiler=${compiler:-clang}
    export JAVA_HOME=$(/usr/libexec/java_home -v 12)
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
echo "Set Java PATH and CPATH"
export PATH=$JAVA_HOME/bin:$PATH
export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/${java_os_name}:$CPATH
echo "Calling make"
make ${target:-daal} ${make_op} \
    PLAT=${platform} \
    COMPILER=${compiler} \
    REQCPU="${CPU_OPTIMIZATIONS}"

exit $?
