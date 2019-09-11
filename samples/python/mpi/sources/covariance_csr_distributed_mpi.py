# file: covariance_csr_distributed_mpi.py
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
# !  Content:
# !    Pyhton sample of sparse variance-covariance matrix computation in the
# !    distributed processing mode
# !
# !*****************************************************************************

#
## <a name"DAAL-SAMPLE-PY-COVARIANCE_CSR_DISTRIBUTED"></a>
## \example covariance_csr_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import covariance
from daal.data_management import OutputDataArchive, InputDataArchive

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

# Input data set parameters
nBlocks = 4
MPI_ROOT = 0

data_dir = jp('data', 'distributed')

datasetFileNames = [
    jp(data_dir, 'covcormoments_csr_1.csv'),
    jp(data_dir, 'covcormoments_csr_2.csv'),
    jp(data_dir, 'covcormoments_csr_3.csv'),
    jp(data_dir, 'covcormoments_csr_4.csv')
]

if __name__ == "__main__":

    comm_size = MPI.COMM_WORLD
    rankId = comm_size.Get_rank()

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataTable = createSparseTable(datasetFileNames[rankId])

    # Create an algorithm to compute a sparse variance-covariance matrix on local nodes
    localAlgorithm = covariance.Distributed(step1Local, method=covariance.fastCSR)

    # Set the input data set to the algorithm
    localAlgorithm.input.set(covariance.data, dataTable)

    # Compute a sparse variance-covariance matrix
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    pres.serialize(dataArch)
    perNodeArchLength = dataArch.getSizeOfArchive()

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    data = comm_size.gather(nodeResults, MPI_ROOT)

    if rankId == MPI_ROOT:
        # Create an algorithm to compute a sparse variance-covariance matrix on the master node
        masterAlgorithm = covariance.Distributed(step2Master, method=covariance.fastCSR)

        for i in range(nBlocks):
            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(data[i])

            dataForStep2FromStep1 = covariance.PartialResult()

            dataForStep2FromStep1.deserialize(dataArch)

            # Set local partial results as input for the master-node algorithm
            masterAlgorithm.input.add(covariance.partialResults, dataForStep2FromStep1)


        # Merge and finalizeCompute a sparse variance-covariance matrix on the master node
        masterAlgorithm.compute()
        result = masterAlgorithm.finalizeCompute() # Retrieve the algorithm results

        # Print the results
        printNumericTable(result.get(covariance.covariance), "Covariance matrix (upper left square 10*10) :", 10, 10)
        printNumericTable(result.get(covariance.mean),       "Mean vector:", 1, 10)
