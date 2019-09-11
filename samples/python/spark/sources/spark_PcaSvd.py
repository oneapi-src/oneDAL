# file: spark_PcaSvd.py
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
#      Python sample of principal component analysis (PCA) using the singular
#      value decomposition (SVD) method in the distributed processing mode
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms import pca
from daal.data_management import OutputDataArchive

from distributed_hdfs_dataset import serializeNumericTable, DistributedHDFSDataSet, deserializeNumericTable

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runPCA(dataRDD):
    partsRDD = computestep1Local(dataRDD)
    return finalizeMergeOnMasterNode(partsRDD)


def computestep1Local(dataRDD):

    def mapper(tup):

        key, homogen_table = tup

        # Create an algorithm to compute PCA decomposition using the SVD method on local nodes
        pcaLocal = pca.Distributed(step1Local, method=pca.svdDense)

        # Set the input data on local nodes
        deserialized_homogen_table = deserializeNumericTable(homogen_table)
        pcaLocal.input.setDataset(pca.data, deserialized_homogen_table)

        # Compute PCA decomposition on local nodes
        pres = pcaLocal.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return dataRDD.map(mapper)


def finalizeMergeOnMasterNode(partsRDD):

    # Create an algorithm to compute PCA decomposition using the SVD method on the master node
    pcaMaster = pca.Distributed(step2Master, method=pca.svdDense)

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for key, pres in parts_list:
        dataArch = OutputDataArchive(pres)
        deserialized_pres = pca.PartialResult(pca.svdDense)
        deserialized_pres.deserialize(dataArch)
        pcaMaster.input.add(pca.partialResults,deserialized_pres)

    # Compute PCA decomposition on the master node
    pcaMaster.compute()

    # Finalize computations and retrieve the results
    res = pcaMaster.finalizeCompute()

    return {'eigenvectors': res.get(pca.eigenvectors), 'eigenvalues': res.get(pca.eigenvalues)}

if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark PCA(SVD)").setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/PcaSvd/data")
    dataRDD = dd.getAsPairRDD(sc)

    # Compute PCA decomposition for dataRDD using the SVD method
    result = runPCA(dataRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('PcaSvd.out', 'w')

    # Print the results
    printNumericTable(result['eigenvalues'], "Eigenvalues:")
    printNumericTable(result['eigenvectors'], "Eigenvectors:")

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
