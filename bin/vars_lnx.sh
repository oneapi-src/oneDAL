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
    echo "Syntax: source $__daal_tmp_script_name [<arch>]"
    echo "Where <arch> is one of:"
    echo "  ia32      - setup environment for IA-32 architecture"
    echo "  intel64   - setup environment for Intel(R) 64 architecture"
    echo ""
    echo "default is intel64"
    echo ""
    echo "If the arguments to the sourced script are ignored (consult docs for"
    echo "your shell) the alternative way to specify target is environment"
    echo "variables COMPILERVARS_ARCHITECTURE or DAALVARS_ARCHITECTURE to pass"
    echo "<arch> to the script."
    echo ""
}

set_daal_env() {
    __daal_tmp_dir=$(command -p cd $(dirname -- "${BASH_SOURCE}")/..; pwd)

    __daal_tmp_script_name="vars.sh"
    __daal_tmp_target_arch="intel64"

    if [ -z "$1" ] ; then
        if [ -n "$DAALVARS_ARCHITECTURE" ] ; then
            __daal_tmp_target_arch="$DAALVARS_ARCHITECTURE"
        elif [ -n "$COMPILERVARS_ARCHITECTURE" ] ; then
            __daal_tmp_target_arch="$COMPILERVARS_ARCHITECTURE"
        fi
    else
        while [ $# -gt 0 ]
        do
            opt="$1"
            case $opt in
                ia32)
                    __daal_tmp_target_arch="ia32"
                    shift
                    ;;
                intel64)
                    __daal_tmp_target_arch="intel64"
                    shift
                    ;;
                *)
                    shift
                    ;;
            esac
        done
    fi

    export DAALROOT=$__daal_tmp_dir
    export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
    export LIBRARY_PATH=$__daal_tmp_dir/lib/${__daal_tmp_target_arch}${LIBRARY_PATH+:${LIBRARY_PATH}}
    export LD_LIBRARY_PATH=$__daal_tmp_dir/lib/${__daal_tmp_target_arch}${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}
    export CLASSPATH=$__daal_tmp_dir/lib/daal.jar${CLASSPATH+:${CLASSPATH}}
}

set_daal_env "$@"
