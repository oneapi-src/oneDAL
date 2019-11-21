#!/bin/bash
#===============================================================================
# Copyright 2017-2019 Intel Corporation
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
##     Intel(R) Data Analytics Acceleration Library samples
##******************************************************************************
# Don't forget to set the env below
# linux:
#  export JAVA_HOME=/usr/intel/pkgs/java/1.7.0.45-64/jre/
#  export PATH=/usr/local/hadoop/bin:/usr/local/spark/bin:/usr/intel/pkgs/java/1.7.0.45-64/bin:$PATH
#  export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/
#  export DAALROOT=${PWD}/../../../daal
#  export CLASSPATH=/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-common-2.5.1.jar:/usr/local/hadoop/share/hadoop/common/hadoop-common-2.5.1.jar:/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.5.1.jar:/usr/local/spark/core/target/spark-core_2.10-1.1.0.jar:/usr/local/spark/mllib/target/spark-mllib_2.10-1.1.0.jar:${DAALROOT}/lib/daal.jar
#  export SCALA_JARS=/tmp/scala-library-2.10.4.jar
#
# macOS*:
#  export JAVA_HOME=$(/usr/libexec/java_home)
#  export PATH=/usr/local/hadoop/bin:/usr/local/spark-1.1.1/bin:$PATH
#  export HADOOP_CONF_DIR=/usr/local/hadoop/etc/hadoop/
#  export DAALROOT=${PWD}/../../../daal
#  export CLASSPATH=/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-common-2.7.0.jar:/usr/local/hadoop/share/hadoop/common/hadoop-common-2.7.0.jar:/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.7.0.jar:/usr/local/spark-1.1.1/core/target/spark-core_2.10-1.1.1.jar:/usr/local/spark-1.1.1/mllib/target/spark-mllib_2.10-1.1.1.jar:${DAALROOT}/lib/daal.jar
#  export SCALA_JARS=/usr/local/scala-2.10.4/lib/scala-library.jar

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
        ia32|intel64)          daal_ia=$1
                               ;;
        help)                  help_message
                               exit 0
                               ;;
        *)                     break
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
export CLASSPATH=${SPARK_HOME}/jars/spark-core_2.11-2.0.0.jar:${SPARK_HOME}/jars/spark-sql_2.11-2.0.0.jar:${SPARK_HOME}/jars/spark-catalyst_2.11-2.0.0.jar:${DAALROOT}/lib/daal.jar:$CLASSPATH
export CLASSPATH=${SCALA_JARS}:$CLASSPATH

# Setting paths by OS
os_name=`uname -s`
if [ "${os_name}" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH=${DAALROOT}/lib/:${TBBROOT}/lib/:$DYLD_LIBRARY_PATH

    TBBLIBS=
    if [ -d ${TBBROOT}/lib ]; then TBBLIBS=${TBBROOT}/lib; fi
    if ! [ -z {$TBBROOT} ] && [ -d ${TBBROOT}/lib ]; then TBBLIBS=${TBBROOT}/lib; fi
    if [ -z ${TBBLIBS} ]; then
        echo Can not find TBB runtimes
        exit 1
    fi

    #Comma-separated list of shared libs
    export SHAREDLIBS=${DAALROOT}/lib/libJavaAPI.dylib,${TBBLIBS}/libtbb.dylib,${TBBLIBS}/libtbbmalloc.dylib
else
    TBBLIBS=
    if [ -d ${TBBROOT}/lib/${daal_ia}/gcc4.8 ]; then TBBLIBS=${TBBROOT}/lib/${daal_ia}/gcc4.8; fi
    if [ -z ${TBBLIBS} ]; then
        echo Can not find TBB runtimes
        exit 1
    fi

    #Comma-separated list of shared libs
    export SHAREDLIBS=${DAALROOT}/lib/${daal_ia}/libJavaAPI.so,${TBBLIBS}/libtbb.so.2,${TBBLIBS}/libtbbmalloc.so.2
fi

# Setting list of Spark samples to process
if [ -z ${Spark_samples_list} ]; then
    source ./daal.lst
fi

for sample in ${Spark_samples_list[@]}; do

    # Creating _results folder
    mkdir -p ./_results/${sample}

    hadoop fs -rm -r /Spark/${sample}/data >> _results/${sample}/${sample}.log 2>&1

    # Creating new folders on HDFS
    hadoop fs -mkdir -p /Spark/${sample}/data > _results/${sample}/${sample}.log 2>&1

    # Putting input data on HDFS
    hadoop fs -put ./data/${sample} /Spark/${sample}/data/ >> _results/${sample}/${sample}.log 2>&1
    hadoop fs -put ./data/${sample}*.csv /Spark/${sample}/data/ >> _results/${sample}/${sample}.log 2>&1

    # Building samples
    javac -d ./_results/${sample} -sourcepath ./ sources/*${sample}.java sources/DistributedHDFSDataSet.java
    cd _results/${sample}

    # Creating jar
    jar -cvfe spark${sample}.jar DAAL.Sample${sample} ./* >> ${sample}.log

    # Running samples. Can be run with "--master yarn-cluster --deploy-mode cluster" as well as with "--master yarn-client"
    cmd="spark-submit --driver-class-path \"${DAALROOT}/lib/daal.jar:${SCALA_JARS}\" --jars ${DAALROOT}/lib/daal.jar --files ${SHAREDLIBS},${DAALROOT}/lib/daal.jar -v --master yarn-cluster --deploy-mode cluster --class DAAL.Sample${sample} spark${sample}.jar"
    echo $cmd > ${sample}.res
    `${cmd} >> ${sample}.res 2>&1`

    # Checking run status
    grepres=`grep 'SUCCEEDED' ${sample}.res 2>/dev/null || true`
    if [ "$grepres" == "" ]; then
        echo -e "`date +'%H:%M:%S'` FAILED\t\t${sample}"
    else
        echo -e "`date +'%H:%M:%S'` PASSED\t\t${sample}"
    fi
    cd ../..

done