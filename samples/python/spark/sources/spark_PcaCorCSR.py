# file: spark_PcaCorCSR.py
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
#      Python sample of principal component analysis (PCA) using the correlation
#      method in the distributed processing mode
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import pca
from daal.algorithms import covariance
from daal.data_management import OutputDataArchive

from distributed_hdfs_dataset import (
    serializeNumericTable, DistributedHDFSDataSet, deserializeCSRNumericTable
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runPCA(dataRDD):
    partsRDD = computestep1Local(dataRDD)
    return finalizeMergeOnMasterNode(partsRDD)


def computestep1Local(dataRDD):

    # public Tuple2<Integer, PartialResult> call(Tuple2<Integer, CSRNumericTable> tup)
    def mapper(tup):
        key, csr_table = tup

        # Create an algorithm to compute PCA decomposition using the correlation method on local nodes
        pcaLocal = pca.Distributed(step1Local, method=pca.correlationDense)

        covarianceSparse = covariance.Distributed(step1Local, method=covariance.fastCSR)
        pcaLocal.parameter.covariance = covarianceSparse

        # Set the input data on local nodes
        deserialized_csr_table = deserializeCSRNumericTable(csr_table)
        pcaLocal.input.setDataset(pca.data, deserialized_csr_table)

        # Compute PCA decomposition on local nodes
        pres = pcaLocal.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return dataRDD.map(mapper)


def finalizeMergeOnMasterNode(partsRDD):

    # Create an algorithm to compute PCA decomposition using the correlation method on the master node
    pcaMaster = pca.Distributed(step2Master, method=pca.correlationDense)

    covarianceSparse = covariance.Distributed(step2Master, method=covariance.fastCSR)
    pcaMaster.parameter.covariance = covarianceSparse

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for key, pres in parts_list:
        dataArch = OutputDataArchive(pres)
        deserialized_pres = pca.PartialResult(pca.correlationDense)
        deserialized_pres.deserialize(dataArch)
        pcaMaster.input.add(pca.partialResults, deserialized_pres)

    # Compute PCA decomposition on the master node
    pcaMaster.compute()

    # Finalize computations and retrieve the results
    res = pcaMaster.finalizeCompute()

    return {'eigenvectors': res.get(pca.eigenvectors), 'eigenvalues': res.get(pca.eigenvalues)}


if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark PCA(COR)").setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/PcaCorCSR/data/")
    dataRDD = dd.getCSRAsPairRDD(sc)

    # Compute PCA decomposition for dataRDD using the correlation method
    result = runPCA(dataRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('PcaCorCSR.out', 'w')

    # Print the results
    printNumericTable(result['eigenvalues'], "Eigenvalues:")
    printNumericTable(result['eigenvectors'], "Eigenvectors:")

    # Restore sdtout
    sys.stdout = stdout

    sc.stop()
