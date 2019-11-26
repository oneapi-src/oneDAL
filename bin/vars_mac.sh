#!/bin/bash
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

daal_help() {
    echo "Syntax: source vars.sh"
}

set_daal_env() {
    __daal_tmp_dir="<INSTALLDIR>"
    __daal_tmp_dir=$__daal_tmp_dir/daal
    if [ ! -d $__daal_tmp_dir ]; then
        __daal_tmp_dir=$(command -p cd $(dirname -- "${BASH_SOURCE}")/..; pwd)
    fi

    export DAALROOT=$__daal_tmp_dir
    export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
    export LIBRARY_PATH=$__daal_tmp_dir/lib${LIBRARY_PATH+:${LIBRARY_PATH}}
    export DYLD_LIBRARY_PATH=$__daal_tmp_dir/lib${DYLD_LIBRARY_PATH+:${DYLD_LIBRARY_PATH}}
    export CLASSPATH=$__daal_tmp_dir/lib/daal.jar${CLASSPATH+:${CLASSPATH}}
}

set_daal_env "$@"
