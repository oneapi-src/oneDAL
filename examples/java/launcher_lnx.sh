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

##  Content:
##     Intel(R) Data Analytics Acceleration Library examples creation and run
##******************************************************************************

help_message() {
    echo "Usage: launcher.sh {arch|help} [rmode] [path_to_javac]"
    echo "arch          - can be ia32 or intel64, optional for building examples."
    echo "rmode         - optional parameter, can be build (for building examples only) or"
    echo "                run (for running examples only)."
    echo "                buildandrun (to perform both)."
    echo "                If not specified build and run are performed."
    echo "path_to_javac - optional parameter."
    echo "                Specify it in case, if you do not want to use default javac"
    echo "help          - print this message"
    echo "Example: launcher.sh ia32 run or launcher.sh intel64 build /export/users/test/jdk1.7/lnx32/jdk1.7.0_67"
}

full_ia=
rmode=
path_to_javac=
first_arg=$1

while [ "$1" != "" ]; do
    case $1 in
        ia32|intel64)          full_ia=$1
                               ;;
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

if [ "${full_ia}" != "ia32" -a "${full_ia}" != "intel64" -a "${rmode}" != "build" ]; then
    echo Bad argument arch = ${first_arg} , must be ia32 or intel64
    help_message
    exit 1
fi

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
Djava_library_path=${DAALROOT}/lib/${full_ia}

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
            if [ "${full_ia}" == "intel64" ]; then memory=4g; else memory=1g; fi

            [ ! -d ${result_folder}/${example_dir} ] && mkdir -p ${result_folder}/${example_dir}

            example_path=com.intel.daal.examples.${example_dir}.${example_name}
            res_path=${result_folder}/${example_dir}/${example_name}.res

            java -Xmx${memory} -Djava.library.path=${Djava_library_path} ${example_path} 2>&1 >${res_path}
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
