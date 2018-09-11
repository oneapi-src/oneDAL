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

##  Content:
##     Intel(R) Data Analytics Acceleration Library examples creation and run
##******************************************************************************

help_message() {
    echo "Usage: launcher.sh {help} [rmode] [path_to_javac]"
    echo "rmode         - optional parameter, can be build (for building examples only) or"
    echo "                run (for running examples only)."
    echo "                buildandrun (to perform both)."
    echo "                If not specified build and run are performed."
    echo "path_to_javac - optional parameter."
    echo "                Specify it in case, if you do not want to use default javac"
    echo "help          - print this message"
    echo "Example: launcher.sh run or launcher.sh build /export/users/test/jdk1.7/lnx32/jdk1.7.0_67"
}

full_ia=intel64
rmode=
path_to_javac=
first_arg=$1

while [ "$1" != "" ]; do
    case $1 in
        build|run|buildandrun) rmode=$1
                               ;;
        help)                  help_message
                               exit 0
                               ;;
        *)                     break
                               ;;
    esac
    shift
done

export CLASSPATH=`pwd`${CLASSPATH+:${CLASSPATH}}
class_path=`pwd`/com/intel/daal/examples

# Setting environment for side javac if the path specified
path_to_javac=$1
if [ "${path_to_javac}" != "" ]; then
    export PATH=${path_to_javac}/bin:${PATH}
fi

# Setting list of Java examples to process
if [ -z ${Java_example_list} ]; then
    source ./daal.lst
fi

# Setting path for JavaAPI library
os_name=`uname -s`
if [ "${os_name}" == "Darwin" ]; then
    Djava_library_path=${DAALROOT}/lib
else
    Djava_library_path=${DAALROOT}/lib/${full_ia}_lin
fi

# Setting a path for result folder to put results of examples in
if [ "${full_ia}"!="" ]; then
    result_folder=./_results/${full_ia}
    if [ -d ${result_folder} ]; then rm -rf ${result_folder}; fi
    mkdir -p ${result_folder}
fi

for example in ${Java_example_list[@]}; do
# Building examples
    if [ "${rmode}" != "run" ]; then
        javac ${class_path}/${example}.java
    fi
# Running examples
    if [ "${rmode}" != "build" ]; then
        arr=(${example//// })
        if [ -z ${arr[2]} ]; then
            example_dir=${arr[0]}
            example_name=${arr[1]}
        else
            example_dir=${arr[0]}/${arr[1]}
            example_name=${arr[2]}
        fi
        if [ -f "${class_path}/${example}.class" ]; then

            [ ! -d ${result_folder}/${example_dir} ] && mkdir -p ${result_folder}/${example_dir}

            example_path=com.intel.daal.examples.${example_dir}.${example_name}
            res_path=${result_folder}/${example_dir}/${example_name}.res

            java -Djava.library.path=${Djava_library_path} ${example_path} 2>&1 >${res_path}
            errcode=$?
            if [ "${errcode}" == "0" ]; then
                 echo -e "`date +'%H:%M:%S'` PASSED\t\t${example_name}"
            else
                 echo -e "`date +'%H:%M:%S'` FAILED\t\t${example_name} with errno ${errcode}"
            fi
        else
            echo -e "`date +'%H:%M:%S'` BUILD FAILED\t\t${example_name}"
        fi
    fi
done
