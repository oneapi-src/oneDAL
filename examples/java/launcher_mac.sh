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
    echo "Usage: launcher.sh {help} [rmode] [path_to_javac]"
    echo "rmode         - optional parameter, can be build (for building examples only) or"
    echo "                run (for running examples only)."
    echo "                buildandrun (to perform both)."
    echo "                If not specified build and run are performed."
    echo "path_to_javac - optional parameter."
    echo "                Specify it in case, if you do not want to use default javac"
    echo "help          - print this message"
    echo "Example: launcher.sh run or launcher.sh build /export/users/test/jdk1.7/mac32e/jdk1.7.0_67"
}

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
Djava_library_path=${DAALROOT}/lib

# Setting a path for result folder to put results of examples in
result_folder=./_results/intel64
if [ -d ${result_folder} ]; then rm -rf ${result_folder}; fi
mkdir -p ${result_folder}

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
