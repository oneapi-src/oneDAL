#!/bin/bash
#===============================================================================
# Copyright 2017-2021 Intel Corporation
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
# Don't forget to set the env below
# linux:
#  export JAVA_HOME=/usr/intel/pkgs/java/1.7.0.45-64/jre/
#  export PATH=/usr/local/hadoop/bin:/usr/local/spark/bin:/usr/intel/pkgs/java/1.7.0.45-64/bin:$PATH
#  export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/
#  export DAALROOT=${PWD}/../../../daal
#  export CLASSPATH=/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-common-2.5.1.jar:/usr/local/hadoop/share/hadoop/common/hadoop-common-2.5.1.jar:/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.5.1.jar:/usr/local/spark/core/target/spark-core_2.10-1.1.0.jar:/usr/local/spark/mllib/target/spark-mllib_2.10-1.1.0.jar:${DAALROOT}/lib/onedal.jar
#  export SCALA_JARS=/tmp/scala-library-2.10.4.jar
#
# macOS*:
#  export JAVA_HOME=$(/usr/libexec/java_home)
#  export PATH=/usr/local/hadoop/bin:/usr/local/spark-1.1.1/bin:$PATH
#  export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/
#  export DAALROOT=${PWD}/../../../daal
#  export CLASSPATH=/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-common-2.7.0.jar:/usr/local/hadoop/share/hadoop/common/hadoop-common-2.7.0.jar:/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.7.0.jar:/usr/local/spark-1.1.1/core/target/spark-core_2.10-1.1.1.jar:/usr/local/spark-1.1.1/mllib/target/spark-mllib_2.10-1.1.1.jar:${DAALROOT}/lib/onedal.jar
#  export SCALA_JARS=/usr/local/scala-2.10.4/lib/scala-library.jar

help_message() {
    echo "Usage: launcher.sh [help]"
    echo "help          - print this message"
    echo "Example: launcher.sh"
}


while [ "$1" != "" ]; do
    case $1 in
        ia32|intel64)          echo "Please switch to new params, 32-bit support deprecated "
                               ;;
        help)                  help_message
                               exit 0
                               ;;
        *)                     break
                               ;;
    esac
    shift
done

# Setting CLASSPATH to build jar
export CLASSPATH=${SPARK_HOME}/jars/spark-core_2.11-2.0.0.jar:${SPARK_HOME}/jars/spark-sql_2.11-2.0.0.jar:${SPARK_HOME}/jars/spark-catalyst_2.11-2.0.0.jar:${SPARK_HOME}/jars/spark-mllib_2.11-2.0.0.jar:${SPARK_HOME}/jars/hadoop-common-2.7.2.jar:${SPARK_HOME}/jars/jackson-annotations-2.6.5.jar:${SPARK_HOME}/jars/breeze_2.11-0.11.2.jar:${SPARK_HOME}/jars/breeze-macros_2.11-0.11.2.jar:${DAALROOT}/lib/onedal.jar:$CLASSPATH
export CLASSPATH=${SCALA_JARS}:$CLASSPATH

# Setting paths by OS
os_name=$(uname -s)
if [ "${os_name}" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${DAALROOT}/lib/:${TBBROOT}/lib/:$DYLD_LIBRARY_PATH
    export LIBJAVAAPI=libJavaAPI.dylib

    TBBLIBS=
    if [ -d "${TBBROOT}/lib" ]; then TBBLIBS=${TBBROOT}/lib; fi
    if [ "${TBBROOT}" ] && [ -d "${TBBROOT}/lib" ]; then TBBLIBS=${TBBROOT}/lib; fi
    if [ -z "${TBBLIBS}" ]; then
        echo Can not find TBB runtimes
        exit 1
    fi

    #Comma-separated list of shared libs
    export SHAREDLIBS=${DAALROOT}/lib/${LIBJAVAAPI}

    if [ -f "${TBBLIBS}/libtbb.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.dylib
    fi
    if [ -f "${TBBLIBS}/libtbb.2.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.2.dylib
    fi
    if [ -f "${TBBLIBS}/libtbb.12.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.12.dylib
    fi

    if [ -f "${TBBLIBS}/libtbbmalloc.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.dylib
    fi
    if [ -f "${TBBLIBS}/libtbbmalloc.2.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.2.dylib
    fi
    if [ -f "${TBBLIBS}/libtbbmalloc.12.dylib" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.12.dylib
    fi
else
    export LIBJAVAAPI=libJavaAPI.so

    TBBLIBS=
    if [ -d "${TBBROOT}/lib/intel64/gcc4.8" ]; then TBBLIBS=${TBBROOT}/lib/intel64/gcc4.8; fi
    if [ -z "${TBBLIBS}" ]; then
        echo Can not find TBB runtimes
        exit 1
    fi

    #Comma-separated list of shared libs
    export SHAREDLIBS=${DAALROOT}/lib/intel64/${LIBJAVAAPI}

    if [ -f "${TBBLIBS}/libtbb.so" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.so
    fi
    if [ -f "${TBBLIBS}/libtbb.so.2" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.so.2
    fi
    if [ -f "${TBBLIBS}/libtbb.so.12" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbb.so.12
    fi

    if [ -f "${TBBLIBS}/libtbbmalloc.so" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.so
    fi
    if [ -f "${TBBLIBS}/libtbbmalloc.so.2" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.so.2
    fi
    if [ -f "${TBBLIBS}/libtbbmalloc.so.12" ]; then
        SHAREDLIBS=${SHAREDLIBS},${TBBLIBS}/libtbbmalloc.so.12
    fi
fi

# Setting list of Spark samples to process
# shellcheck disable=2154,1091
if [ -z "${Spark_samples_list}" ]; then
    source ./daal.lst
fi

for sample in "${Spark_samples_list[@]}"; do

    # Creating _results folder
    mkdir -p "./_results/${sample}"

    # Building samples
    scalac -d "./_results/${sample}/Spark${sample}.jar" -sourcepath ./ sources/*"${sample}".scala

    hadoop fs -rm -r "/Spark/${sample}/data" >> "_results/${sample}/${sample}.log" 2>&1

    # Creating new folders on HDFS
    hadoop fs -mkdir -p "/Spark/${sample}/data" > "_results/${sample}/${sample}.log" 2>&1

    # Putting input data on HDFS
    hadoop fs -put ./data/"${sample}"*.txt "/Spark/${sample}/data/" >> "_results/${sample}/${sample}.log" 2>&1

    # Building samples
    cd "_results/${sample}" || exit 1

    # Running samples. Can be run with "--master yarn-cluster --deploy-mode cluster" as well as with "--master yarn-client"
    cmd=(spark-submit --driver-class-path "${DAALROOT}/lib/onedal.jar:${SCALA_JARS}" --jars "${DAALROOT}/lib/onedal.jar" --files "${SHAREDLIBS},${DAALROOT}/lib/onedal.jar" -v --master yarn-cluster --deploy-mode cluster --class "DAAL.Sample${sample}" "Spark${sample}.jar")
    echo "${cmd[@]}" > "${sample}.res"
    "${cmd[@]}" >> "${sample}.res" 2>&1

    # Checking run status
    grepres=$(grep 'SUCCEEDED' "${sample}.res" 2>/dev/null || true)
    if [ "$grepres" == "" ]; then
        echo -e "$(date +'%H:%M:%S') FAILED\t\t${sample}"
    else
        echo -e "$(date +'%H:%M:%S') PASSED\t\t${sample}"
    fi
    cd ../..

done
