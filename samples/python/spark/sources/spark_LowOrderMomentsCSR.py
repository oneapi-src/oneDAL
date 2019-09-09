# file: spark_LowOrderMomentsCSR.py
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
#      Python sample of computing low order moments in the distributed
#      processing mode.
#
#      Input matrix is stored in the compressed sparse row (CSR) format.
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import low_order_moments

from distributed_hdfs_dataset import (
    DistributedHDFSDataSet, serializeNumericTable, deserializeCSRNumericTable,
    deserializePartialResult
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runMoments(dataRDD):
    partsRDD = computestep1Local(dataRDD)
    return finalizeMergeOnMasterNode(partsRDD)


def computestep1Local(dataRDD):

    def mapper(tup):

        key, val = tup
        # Create an algorithm to compute low order moments on local nodes
        momentsLocal = low_order_moments.Distributed(step1Local, method=low_order_moments.fastCSR)

        # Set the input data on local nodes
        deserialized_val = deserializeCSRNumericTable(val)
        momentsLocal.input.set(low_order_moments.data, deserialized_val)

        # Compute low order moments on local nodes
        pres = momentsLocal.compute()
        serialized_pres = serializeNumericTable(pres)

        return(key, serialized_pres)
    return dataRDD.map(mapper)


def finalizeMergeOnMasterNode(partsRDD):

    # Create an algorithm to compute low order moments on the master node
    momentsMaster = low_order_moments.Distributed(step=step2Master, method=low_order_moments.fastCSR)

    parts_List = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, value in parts_List:
        deserialized_pres = deserializePartialResult(value, low_order_moments)
        momentsMaster.input.add(low_order_moments.partialResults, deserialized_pres)

    # Compute low order moments on the master node
    momentsMaster.compute()

    # Finalize computations and retrieve the results
    return momentsMaster.finalizeCompute()

if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark low_order_moments(sparse)").setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/LowOrderMomentsCSR/data/")
    dataRDD = dd.getCSRAsPairRDD(sc)

    # Compute low order moments for dataRDD
    result = runMoments(dataRDD)

    # Print the results
    minimum = result.get(low_order_moments.minimum)
    maximum = result.get(low_order_moments.maximum)
    sum = result.get(low_order_moments.sum)
    sumSquares = result.get(low_order_moments.sumSquares)
    sumSquaresCentered = result.get(low_order_moments.sumSquaresCentered)
    mean = result.get(low_order_moments.mean)
    secondOrderRawMoment = result.get(low_order_moments.secondOrderRawMoment)
    variance = result.get(low_order_moments.variance)
    standardDeviation = result.get(low_order_moments.standardDeviation)
    variation = result.get(low_order_moments.variation)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('LowOrderMomentsCSR.out', 'w')

    print("Low order moments:")
    printNumericTable(minimum, "Min:")
    printNumericTable(maximum, "Max:")
    printNumericTable(sum, "Sum:")
    printNumericTable(sumSquares, "SumSquares:")
    printNumericTable(sumSquaresCentered, "SumSquaredDiffFromMean:")
    printNumericTable(mean, "Mean:")
    printNumericTable(secondOrderRawMoment, "SecondOrderRawMoment:")
    printNumericTable(variance, "Variance:")
    printNumericTable(standardDeviation, "StandartDeviation:")
    printNumericTable(variation, "Variation:")

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
