# file: spark_CovarianceDense.py
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

#
#  Content:
#     Python sample of dense variance-covariance matrix computation
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import covariance
from daal.data_management import OutputDataArchive

from distributed_hdfs_dataset import DistributedHDFSDataSet
from distributed_hdfs_dataset import serializeNumericTable
from distributed_hdfs_dataset import deserializeNumericTable

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runCovariance(dataRDD):
    partsRDD = computestep1Local(dataRDD)
    return finalizeMergeOnMasterNode(partsRDD)


def computestep1Local(dataRDD):

    def mapper(tup):

        key, val = tup

        # Create an algorithm to compute a dense variance-covariance matrix on local nodes
        covarianceLocal = covariance.Distributed(step=step1Local, method=covariance.defaultDense)

        # Set the input data on local nodes
        deserialized_val = deserializeNumericTable(val)
        covarianceLocal.input.set(covariance.data, deserialized_val)

        # Compute a dense variance-covariance matrix on local nodes
        pres = covarianceLocal.compute()

        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)

    return dataRDD.map(mapper)


def finalizeMergeOnMasterNode(partsRDD):

    # Create an algorithm to compute a dense variance-covariance matrix on the master node
    covarianceMaster = covariance.Distributed(step=step2Master, method=covariance.defaultDense)

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, val in parts_list:
        dataArch = OutputDataArchive(val)
        deserialized_val = covariance.PartialResult()
        deserialized_val.deserialize(dataArch)
        covarianceMaster.input.add(covariance.partialResults, deserialized_val)

    # Compute a dense variance-covariance matrix on the master node
    covarianceMaster.compute()

    # Finalize computations and retrieve the results
    res = covarianceMaster.finalizeCompute()

    result = {}
    result['covariance'] = res.get(covariance.covariance)
    result['mean'] = res.get(covariance.mean)

    return result

if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark covariance(dense)").setMaster("local[4]"))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/CovarianceDense/data/")
    dataRDD = dd.getAsPairRDD(sc)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('CovarianceDense.out', 'w')

    # Compute a dense variance-covariance matrix for dataRDD
    final_result = runCovariance(dataRDD)

    # Print the results
    printNumericTable(final_result['covariance'], "Covariance:", interval=9)
    printNumericTable(final_result['mean'], "Mean:", interval=9)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
