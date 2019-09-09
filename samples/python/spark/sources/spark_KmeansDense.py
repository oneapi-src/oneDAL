# file: spark_KmeansDense.py
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
#      Python sample of K-Means clustering in the distributed processing mode
#
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import kmeans
from daal.algorithms.kmeans import init

from distributed_hdfs_dataset import serializeNumericTable, DistributedHDFSDataSet, deserializePartialResult, deserializeNumericTable

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


nBlocks = 1
nClusters = 20
nIterations = 5
nVectorsInBlock = 10000


def runKmeans(dataRDD):

    partsRDD = computeInitLocal(dataRDD)
    centroids = computeInitMaster(partsRDD)

    for it in range(nIterations):
        serialized_centroids = serializeNumericTable(centroids)
        partsRDDcompute = computeLocal(dataRDD, serialized_centroids)
        centroids = computeMaster(partsRDDcompute)

    return centroids


def computeInitLocal(dataRDD):

    def mapper(tup):
        key, val = tup
        # Create an algorithm to initialize the K-Means algorithm on local nodes
        kmeansLocalInit = init.Distributed(step1Local,
                                           nClusters,
                                           nBlocks * nVectorsInBlock,
                                           nVectorsInBlock * key,
                                           method=init.randomDense)

        # Set the input data on local nodes
        deserialized_val = deserializeNumericTable(val)
        kmeansLocalInit.input.set(init.data, deserialized_val)

        # Initialize the K-Means algorithm on local nodes
        pres = kmeansLocalInit.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return dataRDD.map(mapper)


def computeInitMaster(partsRDD):

    # Create an algorithm to compute k-means on the master node
    kmeansMasterInit = init.Distributed(step2Master, nClusters, method=init.randomDense)

    partsList = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, value in partsList:
        deserialized_pres = deserializePartialResult(value, init)
        kmeansMasterInit.input.add(init.partialResults, deserialized_pres)

    # Compute k-means on the master node
    kmeansMasterInit.compute()

    # Finalize computations and retrieve the results
    initResult = kmeansMasterInit.finalizeCompute()

    return initResult.get(init.centroids)


def computeLocal(dataRDD, centroids):

    def mapper(tup):

        key, val = tup

        # Create an algorithm to compute k-means on local nodes
        kmeansLocal = kmeans.Distributed(step1Local, nClusters, method=kmeans.defaultDense)

        # Set the input data on local nodes
        deserialized_val = deserializeNumericTable(val)
        deserialized_centroids = deserializeNumericTable(centroids)
        kmeansLocal.input.set(kmeans.data, deserialized_val)
        kmeansLocal.input.set(kmeans.inputCentroids, deserialized_centroids)

        # Compute k-means on local nodes
        pres = kmeansLocal.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return dataRDD.map(mapper)


def computeMaster(partsRDDcompute):

    # Create an algorithm to compute k-means on the master node
    kmeansMaster = kmeans.Distributed(step2Master, nClusters, method=kmeans.defaultDense)

    parts_List = partsRDDcompute.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, value in parts_List:
        deserialized_pres = deserializePartialResult(value, kmeans)
        kmeansMaster.input.add(kmeans.partialResults, deserialized_pres)

    # Compute k-means on the master node
    kmeansMaster.compute()

    # Finalize computations and retrieve the results
    res = kmeansMaster.finalizeCompute()

    return res.get(kmeans.centroids)

if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark Kmeans").setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/KmeansDense/data/")
    dataRDD = dd.getAsPairRDD(sc)

    # Compute k-means for dataRDD
    result = runKmeans(dataRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('KmeansDense.out', 'w')

    # Print the results
    printNumericTable(result, "First 10 dimensions of centroids:", 20, 10)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
