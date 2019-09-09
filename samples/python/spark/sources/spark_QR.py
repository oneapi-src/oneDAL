# file: spark_QR.py
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
#     Python sample of computing QR decomposition in the distributed processing
#     mode
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext
from pyspark.conf import SparkConf

from daal import step1Local, step2Master, step3Local
from daal.algorithms import qr

from distributed_hdfs_dataset import (
    DistributedHDFSDataSet, deserializeNumericTable, serializeNumericTable, deserializeDataCollection
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runQR(dataRDD, sc):

    partsRDD = computeStep1Local(dataRDD)
    res = computeStep2Master(partsRDD['1for2'])
    q = computeStep3Local(partsRDD['1for3'], res['from2_for3'])

    return {'Q': q, 'R': res['ntR']}


def computeStep1Local(dataRDD):
    # Create an RDD containing partial results for steps 2 and 3
    def mapper(tup):

        key, homogen_table = tup

        # Create an algorithm to compute QR decomposition on local nodes
        qrStep1Local = qr.Distributed(step1Local, method=qr.defaultDense)
        deserialized_homogen_table = deserializeNumericTable(homogen_table)
        qrStep1Local.input.set(qr.data, deserialized_homogen_table)

        # Compute QR decomposition in step 1
        pres = qrStep1Local.compute()
        dataFromStep1ForStep2 = pres.get(qr.outputOfStep1ForStep2)
        serialized_1for2 = serializeNumericTable(dataFromStep1ForStep2)
        dataFromStep1ForStep3 = pres.get(qr.outputOfStep1ForStep3)
        serialized_1for3 = serializeNumericTable(dataFromStep1ForStep3)

        return (key, (serialized_1for2, serialized_1for3))

    dataFromStep1_RDD = dataRDD.map(mapper)

    def mapper_for_step3(tup):
        key, data_collections = tup
        dc1, dc2 = data_collections

        return (key, dc2)

    # Extract partial results for step 3
    dataFromStep1ForStep3_RDD = dataFromStep1_RDD.map(mapper_for_step3)

    def mapper_for_step2(tup):
        key, data_collections = tup
        dc1, dc2 = data_collections
        return (key, dc1)

    # Extract partial results for step 2
    dataFromStep1ForStep2_RDD = dataFromStep1_RDD.map(mapper_for_step2)

    return {'1for3': dataFromStep1ForStep3_RDD, '1for2':dataFromStep1ForStep2_RDD}


def computeStep2Master(dataFromStep1ForStep2_RDD):

    nBlocks = int(dataFromStep1ForStep2_RDD.count())

    dataFromStep1ForStep2_list = dataFromStep1ForStep2_RDD.collect()

    # Create an algorithm to compute QR decomposition on the master node
    qrStep2Master = qr.Distributed(step2Master, method=qr.defaultDense)

    for key, collection in dataFromStep1ForStep2_list:
        deserialized_collection = deserializeDataCollection(collection)
        qrStep2Master.input.add(qr.inputOfStep2FromStep1, key, deserialized_collection)

    # Compute QR decomposition in step 2
    pres = qrStep2Master.compute()

    inputForStep3FromStep2 = pres.getCollection(qr.outputOfStep2ForStep3)

    tup_list = []
    for key, collection in dataFromStep1ForStep2_list:
        dc = inputForStep3FromStep2[key]
        serialized_dc = serializeNumericTable(dc)
        tup_list.append((key, serialized_dc))

    # Make PairRDD from the list
    dataFromStep2ForStep3_RDD = sc.parallelize(tup_list, nBlocks)

    res = qrStep2Master.finalizeCompute()

    ntR = res.get(qr.matrixR)

    return {'ntR': ntR, 'from2_for3': dataFromStep2ForStep3_RDD}


def computeStep3Local(dataFromStep1ForStep3_RDD, dataFromStep2ForStep3_RDD):
    # Group partial results from steps 1 and 2
    dataForStep3_RDD = dataFromStep1ForStep3_RDD.cogroup(dataFromStep2ForStep3_RDD)

    def mapper(tup):
        key, collections = tup
        dc1, dc2 = collections

        ntQPi = dc1.__iter__().next()
        deserialized_ntQPi = deserializeDataCollection(ntQPi)
        ntPi = dc2.__iter__().next()
        deserialized_ntPi = deserializeDataCollection(ntPi)

        # Create an algorithm to compute QR decomposition on the master node
        qrStep3Local = qr.Distributed(step3Local, method=qr.defaultDense)
        qrStep3Local.input.set(qr.inputOfStep3FromStep1, deserialized_ntQPi)
        qrStep3Local.input.set(qr.inputOfStep3FromStep2, deserialized_ntPi)

        # Compute QR decomposition in step 3
        qrStep3Local.compute()
        result = qrStep3Local.finalizeCompute()

        Qi = result.get(qr.matrixQ)
        serialized_Qi = serializeNumericTable(Qi)

        return (key, serialized_Qi)

    return dataForStep3_RDD.map(mapper)


if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark QR").setMaster('local[4]'))

    # Read from the distributed HDFS data set at a specified path
    dd = DistributedHDFSDataSet("/Spark/QR/data/")
    dataRDD = dd.getAsPairRDD(sc)

    # Compute QR decomposition for dataRDD
    result = runQR(dataRDD, sc)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('QR.out', 'w')

    # Print the results
    ntRPList = result['Q'].collect()
    for key, table in ntRPList:
        deserialized_table = deserializeNumericTable(table)
        printNumericTable(deserialized_table, "Q (2 first vectors from node #{}):".format(key), 2)

    printNumericTable(result['R'], "R:")

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
