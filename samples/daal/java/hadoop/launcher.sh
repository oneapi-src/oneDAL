#!/bin/bash
#===============================================================================
# Copyright 2017 Intel Corporation
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
##     Intel(R) oneAPI Data Analytics Library samples
##******************************************************************************

help_message() {
    echo "Usage: launcher.sh [help]"
    echo "help          - print this message"
    echo "Example: launcher.sh"
}

echo "Deprecation Notice: The Java interfaces are deprecated in the oneDAL library and may no longer be supported in future releases."

while [ "$1" != "" ]; do
    case $1 in
        ia32|intel64) echo "Please switch to new params, 32-bit support deprecated "
                      ;;
        help)         help_message
                      exit 0
                      ;;
        *)            break
                      ;;
    esac
    shift
done

# Setting CLASSPATH to build jar
export CLASSPATH=${DAALROOT}/lib/onedal.jar:${SCALA_JARS}:$CLASSPATH

# Creating _results folder
result_folder=$(command -p cd "$(dirname -- "${BASH_SOURCE[0]}")"; pwd)/_results/
if [ -d "${result_folder}" ]; then rm -rf "${result_folder}"; fi
mkdir -p "${result_folder}"

hdfs dfs -mkdir -p /Hadoop/Libraries  >  "${result_folder}"/hdfs.log 2>&1

# Comma-separated list of shared libs
os_name=$(uname)

export LIBJAVAAPI=libJavaAPI.so
export LIBTBB=
export LIBTBBMALLOC=

TBBLIBS=
if [ -d "${TBBROOT}/lib/intel64/gcc4.8" ]; then TBBLIBS=${TBBROOT}/lib/intel64/gcc4.8; fi
if [ -z "${TBBLIBS}" ]; then
    echo Can not find TBB runtimes
    exit 1
fi

if [ -f "${TBBLIBS}/libtbb.so.2" ]; then
    export LIBTBB=libtbb.so.2
elif [ -f "${TBBLIBS}/libtbb.so.12" ]; then
    export LIBTBB=libtbb.so.12
else 
    echo Can not find libtbb.so
    exit 1
fi

if [ -f "${TBBLIBS}/libtbbmalloc.so.2" ]; then
    export LIBTBBMALLOC=libtbbmalloc.so.2
elif [ -f "${TBBLIBS}/libtbbmalloc.so.12" ]; then
    export LIBTBBMALLOC=libtbbmalloc.so.12
else 
    echo Can not find libtbbmalloc.so
    exit 1
fi

hdfs dfs -put -f "${DAALROOT}/lib/intel64/${LIBJAVAAPI}" "${TBBLIBS}/${LIBTBB}" "${TBBLIBS}/${LIBTBBMALLOC}" /Hadoop/Libraries/ >> "${result_folder}/hdfs.log" 2>&1

# Setting envs
export LIBJARS=${DAALROOT}/lib/onedal.jar
export CLASSPATH=${LIBJARS}:${CLASSPATH}
export HADOOP_CLASSPATH=${LIBJARS}

# Setting list of Spark samples to process
# shellcheck disable=2154,1091
if [ -z "${Hadoop_samples_list}" ]; then
    source ./daal.lst
fi

for sample in "${Hadoop_samples_list[@]}"; do

    results=${result_folder}/${sample}/
    mkdir "${results}"

    # Delete output folder if it exists
    # shellcheck disable=2129
    hdfs dfs -rm -r -f "/Hadoop/${sample}"       >> "${results}/${sample}.log" 2>&1

    # Create required folders on HDFS
    hdfs dfs -mkdir -p "/Hadoop/${sample}/input" >> "${results}/${sample}.log" 2>&1
    hdfs dfs -mkdir -p "/Hadoop/${sample}/data"  >> "${results}/${sample}.log" 2>&1

    # Copy datasets to HDFS
    hdfs dfs -put ./data/"${sample}"*.csv "/Hadoop/${sample}/data/" >> "${results}/${sample}.log" 2>&1

    cd sources || exit 1

    # Copy file with the dataset names to the input folder
    hdfs dfs -put "${sample}_filelist.txt" "/Hadoop/${sample}/input/" >> "${results}/${sample}.log" 2>&1

    # Building the sample
    mkdir -p ../build
    javac -d ./../build/ -sourcepath ./ ./"${sample}"*.java ./WriteableData.java >> "${results}/${sample}.log" 2>&1

    # Running the sample
    cd ../build/ || exit 1
    jar -cvfe "${sample}.jar" "DAAL.${sample}" ./DAAL/"${sample}"* ./DAAL/WriteableData.class >> "${results}/${sample}.log" 2>&1

    cmd="hadoop jar ${sample}.jar -libjars ${LIBJARS} /Hadoop/${sample}/input /Hadoop/${sample}/Results"
    echo "${cmd}" > "${sample}.log"
    ${cmd} >> "${results}/${sample}.log" 2>&1

    hdfs dfs -ls "/Hadoop/${sample}/Results" >> "${results}/${sample}.log" 2>&1

    # Checking run status
    grepsuccess=$(grep '/Results/_SUCCESS' "${results}/${sample}.log" 2>/dev/null)
    greperror=$(grep 'Error:' "${results}/${sample}.log" 2>/dev/null)
    if [ "$grepsuccess" == "" ] || [ "$greperror" != "" ]; then
        echo -e "$(date +'%H:%M:%S') FAILED\t\t${sample}"
    else
        echo -e "$(date +'%H:%M:%S') PASSED\t\t${sample}"
    fi

    hdfs dfs -get /Hadoop/"${sample}"/Results/* "${results}" >> "${results}/${sample}.log" 2>&1

    cd ../
done
