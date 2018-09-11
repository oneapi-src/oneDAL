#!/bin/bash
#===============================================================================
# Copyright 2014-2018 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

daal_help() {
    echo "Syntax: source daalvars.sh"
}

set_daal_env() {
    __daal_tmp_dir="<INSTALLDIR>"
    __daal_tmp_dir=$__daal_tmp_dir/daal
    if [ ! -d $__daal_tmp_dir ]; then
        __daal_tmp_dir=$(command -p cd $(dirname -- "${BASH_SOURCE}")/..; pwd)
    fi

    export DAALROOT=$__daal_tmp_dir
    export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
    if [ -z "${TBBROOT}" ]; then
        export LIBRARY_PATH=$__daal_tmp_dir/lib:$__daal_tmp_dir/../tbb/lib${LIBRARY_PATH+:${LIBRARY_PATH}}
        export DYLD_LIBRARY_PATH=$__daal_tmp_dir/lib:$__daal_tmp_dir/../tbb/lib${DYLD_LIBRARY_PATH+:${DYLD_LIBRARY_PATH}}
    else
        export LIBRARY_PATH=$__daal_tmp_dir/lib${LIBRARY_PATH+:${LIBRARY_PATH}}
        export DYLD_LIBRARY_PATH=$__daal_tmp_dir/lib${DYLD_LIBRARY_PATH+:${DYLD_LIBRARY_PATH}}
    fi
    export CLASSPATH=$__daal_tmp_dir/lib/daal.jar${CLASSPATH+:${CLASSPATH}}
}

set_daal_env "$@"
