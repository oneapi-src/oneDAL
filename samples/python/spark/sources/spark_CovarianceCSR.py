# file: spark_CovarianceCSR.py
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

#
#  Content:
#     Python sample of sparse variance-covariance matrix computation
#
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import covariance

from distributed_hdfs_dataset import (
    DistributedHDFSDataSet, serializeNumericTable, deserializeCSRNumericTable,
    deserializePartialResult
)

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

        # Create an algorithm to compute a sparse variance-covariance matrix on local nodes
        covarianceLocal = covariance.Distributed(step=step1Local, method=covariance.fastCSR)

        # Deserialize the data into a CSRNumericTable
        deserialized_val = deserializeCSRNumericTable(val)

        # Set the input data on local nodes
        covarianceLocal.input.set(covariance.data, deserialized_val)

        # Compute a sparse variance-covariance matrix on local nodes
        pres = covarianceLocal.compute()

        # Serialize the result
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)

    return dataRDD.map(mapper)


def finalizeMergeOnMasterNode(partsRDD):

    # Create an algorithm to compute a sparse variance-covariance matrix on the master node
    covarianceMaster = covariance.Distributed(step=step2Master, method=covariance.fastCSR)

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, value in parts_list:
        deserialized_value = deserializePartialResult(value, covariance)
        covarianceMaster.input.add(covariance.partialResults, deserialized_value)

    # Compute a sparse variance-covariance matrix on the master node
    covarianceMaster.compute()

    # Finalize computations and retrieve the results
    res = covarianceMaster.finalizeCompute()

    result = {}
    result['covariance'] = res.get(covariance.covariance)
    result['mean'] = res.get(covariance.mean)

    return result

if __name__ == "__main__":

    # Create JavaSparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark covariance(CSR)").setMaster("local[4]"))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/CovarianceCSR/data/")
    dataRDD = dd.getCSRAsPairRDD(sc)

    # Compute a sparse variance-covariance matrix for dataRDD
    final_result = runCovariance(dataRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('CovarianceCSR.out', 'w')

    # Print the results
    printNumericTable(final_result['covariance'], "Covariance matrix (upper left square 10*10) :", 10, 10, 9)
    printNumericTable(final_result['mean'], "Mean vector:", 1, 10, 9)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
