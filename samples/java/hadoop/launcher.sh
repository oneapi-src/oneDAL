#!/bin/bash
#===============================================================================
# Copyright 2017-2019 Intel Corporation.
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
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
#===============================================================================

##  Content:
##     Intel(R) Data Analytics Acceleration Library samples
##******************************************************************************

help_message() {
    echo "Usage: launcher.sh {arch|help}"
    echo "arch          - can be ia32 or intel64, optional for building examples."
    echo "help          - print this message"
    echo "Example: launcher.sh ia32 or launcher.sh intel64"
}

daal_ia=
first_arg=$1

while [ "$1" != "" ]; do
    case $1 in
        ia32|intel64) daal_ia=$1
                      ;;
        help)         help_message
                      exit 0
                      ;;
        *)            break
                      ;;
    esac
    shift
done

if [ "${daal_ia}" != "ia32" -a "${daal_ia}" != "intel64" ]; then
    echo Bad argument arch = ${first_arg} , must be ia32 or intel64
    help_message
    exit 1
fi

# Setting CLASSPATH to build jar
export CLASSPATH=${DAALROOT}/lib/daal.jar:${SCALA_JARS}:$CLASSPATH

# Creating _results folder
result_folder=$(command -p cd $(dirname -- "${BASH_SOURCE}"); pwd)/_results/
if [ -d ${result_folder} ]; then rm -rf ${result_folder}; fi
mkdir -p ${result_folder}

hdfs dfs -mkdir -p /Hadoop/Libraries                                                        >  ${result_folder}/hdfs.log 2>&1

# Comma-separated list of shared libs
os_name=`uname`
if [ "${os_name}" == "Linux" ]; then
    export LIBJAVAAPI=libJavaAPI.so
    export LIBTBB=libtbb.so.2
    export LIBTBBMALLOC=libtbbmalloc.so.2
    hdfs dfs -put -f ${DAALROOT}/lib/${daal_ia}_lin/${LIBJAVAAPI} ${DAALROOT}/../tbb/lib/${daal_ia}_lin/gcc4.4/${LIBTBB} ${DAALROOT}/../tbb/lib/${daal_ia}_lin/gcc4.4/${LIBTBBMALLOC} /Hadoop/Libraries/   >> ${result_folder}/hdfs.log 2>&1
elif [ "${os_name}" == "Darwin" ]; then
    export LIBJAVAAPI=libJavaAPI.dylib
    export LIBTBB=libtbb.dylib
    export LIBTBBMALLOC=libtbbmalloc.dylib
    hdfs dfs -put -f ${DAALROOT}/lib/${LIBJAVAAPI} ${DAALROOT}/../tbb/lib/${LIBTBB} ${DAALROOT}/../tbb/lib/${LIBTBBMALLOC} /Hadoop/Libraries/ >> ${result_folder}/hdfs.log 2>&1
fi

# Setting envs
export LIBJARS=${DAALROOT}/lib/daal.jar
export CLASSPATH=${LIBJARS}:${CLASSPATH}
export HADOOP_CLASSPATH=${LIBJARS}

# Setting list of Spark samples to process
if [ -z ${Hadoop_samples_list} ]; then
    source ./daal.lst
fi

for sample in ${Hadoop_samples_list[@]}; do

    results=${result_folder}/${sample}/
    mkdir ${results}

    # Delete output folder if it exists
    hdfs dfs -rm -r -f /Hadoop/${sample}       >> ${results}/${sample}.log 2>&1

    # Create required folders on HDFS
    hdfs dfs -mkdir -p /Hadoop/${sample}/input >> ${results}/${sample}.log 2>&1
    hdfs dfs -mkdir -p /Hadoop/${sample}/data  >> ${results}/${sample}.log 2>&1

    # Copy datasets to HDFS
    hdfs dfs -put ./data/${sample}*.csv /Hadoop/${sample}/data/ >> ${results}/${sample}.log 2>&1

    cd sources

    # Copy file with the dataset names to the input folder
    hdfs dfs -put ${sample}_filelist.txt /Hadoop/${sample}/input/ >> ${results}/${sample}.log 2>&1

    # Building the sample
    mkdir -p ../build
    javac -d ./../build/ -sourcepath ./ ./${sample}*.java ./WriteableData.java >> ${results}/${sample}.log 2>&1

    # Running the sample
    cd ../build/
    jar -cvfe ${sample}.jar DAAL.${sample} ./DAAL/${sample}* ./DAAL/WriteableData.class >> ${results}/${sample}.log 2>&1

    cmd="hadoop jar ${sample}.jar -libjars ${LIBJARS} /Hadoop/${sample}/input /Hadoop/${sample}/Results"
    echo $cmd > ${sample}.log
    `${cmd} >> ${results}/${sample}.log 2>&1`

    hdfs dfs -ls /Hadoop/${sample}/Results >> ${results}/${sample}.log 2>&1

    # Checking run status
    grepsuccess=`grep '/Results/_SUCCESS' ${results}/${sample}.log 2>/dev/null`
    greperror=`grep 'Error:' ${results}/${sample}.log 2>/dev/null`
    if [ "$grepsuccess" == "" ] || [ "$greperror" != "" ]; then
        echo -e "`date +'%H:%M:%S'` FAILED\t\t${sample}"
    else
        echo -e "`date +'%H:%M:%S'` PASSED\t\t${sample}"
    fi

    hdfs dfs -get /Hadoop/${sample}/Results/* ${results} >> ${results}/${sample}.log 2>&1

    cd ../
done