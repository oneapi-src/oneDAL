# file: spark_Svd.py
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
#     Python sample of computing singular value decomposition (SVD) in the
#     distributed processing mode
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext
from pyspark.conf import SparkConf

from daal import step1Local, step2Master, step3Local
from daal.algorithms import svd
from daal.data_management import OutputDataArchive, DataCollection

from distributed_hdfs_dataset import (
    DistributedHDFSDataSet, deserializeNumericTable, serializeNumericTable, deserializeDataCollection
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runSVD(dataRDD):

    partsRDD = computeStep1Local(dataRDD)
    result = computeStep2Master(partsRDD['1for2_RDD'])
    step3_result = computeStep3Local(partsRDD['1for3_RDD'], result['from2_for3'])
    return {'Sigma': result['Sigma'], 'V': result['V'], 'U': step3_result}


def computeStep1Local(dataRDD):

    def mapper(tup):
        key, homogen_table = tup

        # Create an algorithm to compute SVD on local nodes
        svdStep1Local = svd.Distributed(step1Local)

        deserialized_homogen_table = deserializeNumericTable(homogen_table)
        svdStep1Local.input.set(svd.data, deserialized_homogen_table)

        # Compute SVD in step 1
        pres = svdStep1Local.compute()
        dataFromStep1ForStep2 = pres.get(svd.outputOfStep1ForStep2)
        serialized_data_1for2 = serializeNumericTable(dataFromStep1ForStep2)
        dataFromStep1ForStep3 = pres.get(svd.outputOfStep1ForStep3)
        serialized_data_1for3 = serializeNumericTable(dataFromStep1ForStep3)

        return (key, (serialized_data_1for2, serialized_data_1for3))

    # Create an RDD containing partial results for steps 2 and 3
    dataFromStep1_RDD = dataRDD.map(mapper).cache()

    # Extract partial results for step 3
    dataFromStep1ForStep3_RDD = dataFromStep1_RDD.map(lambda t: (t[0], t[1][1]))

    # Extract partial results for step 2
    dataFromStep1ForStep2_RDD = dataFromStep1_RDD.map(lambda t: (t[0], t[1][0]))

    return {'1for3_RDD': dataFromStep1ForStep3_RDD, '1for2_RDD': dataFromStep1ForStep2_RDD}


def computeStep2Master(dataFromStep1ForStep2_RDD):

    nBlocks = int(dataFromStep1ForStep2_RDD.count())

    dataFromStep1ForStep2_list = dataFromStep1ForStep2_RDD.collect()

    # Create an algorithm to compute SVD on the master node
    svdStep2Master = svd.Distributed(step2Master, method=svd.defaultDense)

    for key, data_collection in dataFromStep1ForStep2_list:
        dataArch = OutputDataArchive(data_collection)
        deserialized_data_collection = DataCollection()
        deserialized_data_collection.deserialize(dataArch)
        svdStep2Master.input.add(svd.inputOfStep2FromStep1, key, deserialized_data_collection)

    # Compute SVD in step 2
    pres = svdStep2Master.compute()

    inputForStep3FromStep2 = pres.getCollection(svd.outputOfStep2ForStep3)

    data_list = []
    for key, _ in dataFromStep1ForStep2_list:
        dc = inputForStep3FromStep2[key]
        serialized_dc = serializeNumericTable(dc)
        data_list.append((key, serialized_dc))

    # Make PairRDD from the list
    dataFromStep2ForStep3_RDD = sc.parallelize(data_list, nBlocks)

    res = svdStep2Master.finalizeCompute()

    result = {
        'Sigma': res.get(svd.singularValues),
        'V': res.get(svd.rightSingularMatrix),
        'from2_for3': dataFromStep2ForStep3_RDD
    }

    return result


def computeStep3Local(dataFromStep1ForStep3_RDD, dataFromStep2ForStep3_RDD):

    # Group partial results from steps 1 and 2
    dataForStep3_RDD = dataFromStep1ForStep3_RDD.cogroup(dataFromStep2ForStep3_RDD)

    def mapper(tup):
        # Tuple2<Integer, Tuple2<Iterable<DataCollection>, Iterable<DataCollection>>> tup)
        key, val = tup
        dc1, dc2 = val

        ntQPi = dc1.__iter__().next()
        deserialized_ntQPi = deserializeDataCollection(ntQPi)
        ntPi = dc2.__iter__().next()
        deserialized_ntPi = deserializeDataCollection(ntPi)

        # Create an algorithm to compute SVD on the master node
        svdStep3Local = svd.Distributed(step3Local, method=svd.defaultDense)
        svdStep3Local.input.set(svd.inputOfStep3FromStep1, deserialized_ntQPi)
        svdStep3Local.input.set(svd.inputOfStep3FromStep2, deserialized_ntPi)

        # Compute SVD in step 3
        svdStep3Local.compute()
        res = svdStep3Local.finalizeCompute()

        Ui = res.get(svd.leftSingularMatrix)
        serialized_Ui = serializeNumericTable(Ui)

        return (key, serialized_Ui)

    return dataForStep3_RDD.map(mapper)


if __name__ == '__main__':

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName('Spark SVD').setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/Svd/data/")
    dataRDD = dd.getAsPairRDD(sc)

    # Compute SVD decomposition for dataRDD
    res = runSVD(dataRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('Svd.out', 'w')

    # Print the results
    ntRPList = res['U'].collect()

    for num, table in ntRPList:
        deserialized_table = deserializeNumericTable(table)
        printNumericTable(deserialized_table, "U (2 first vectors from node #{}):".format(num), 2)

    printNumericTable(res['Sigma'], "Sigma:")
    printNumericTable(res['V'], "V:")

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
