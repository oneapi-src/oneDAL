# file: pca_svd_distributed_mpi.py
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
# !    Python sample of principal component analysis (PCA) using the singular value
# !    decomposition (SVD) method in the distributed processing mode
# !
# !*****************************************************************************

#
## <a name="DAAL-SAMPLE-PY-PCA_SVD_DISTRIBUTED"></a>
## \example pca_svd_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import pca
from daal.data_management import OutputDataArchive, InputDataArchive, FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DATA_PREFIX = jp('data', 'distributed')

# Input data set parameters
nBlocks = 4
MPI_ROOT = 0

datasetFileNames = [
    jp(DATA_PREFIX, 'pca_normalized_1.csv'),
    jp(DATA_PREFIX, 'pca_normalized_2.csv'),
    jp(DATA_PREFIX, 'pca_normalized_3.csv'),
    jp(DATA_PREFIX, 'pca_normalized_4.csv')
]

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileNames[rankId],
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm for principal component analysis using the SVD method on local nodes
    localAlgorithm = pca.Distributed(step1Local, method=pca.svdDense)

    # Set the input data set to the algorithm
    localAlgorithm.input.setDataset(pca.data, dataSource.getNumericTable())

    # Compute PCA decomposition
    # PartialResult_SvdDense class from pca
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    pres.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:
        # Create an algorithm for principal component analysis using the SVD method on the master node
        masterAlgorithm = pca.Distributed(step2Master, method=pca.svdDense)

        for i in range(nBlocks):
            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(serializedData[i])

            dataForStep2FromStep1 = pca.PartialResult(pca.svdDense)
            dataForStep2FromStep1.deserialize(dataArch)

            # Set local partial results as input for the master-node algorithm
            masterAlgorithm.input.add(pca.partialResults, dataForStep2FromStep1)

        # Merge and finalizeCompute PCA decomposition on the master node
        masterAlgorithm.compute()
        res = masterAlgorithm.finalizeCompute()

        # Print the results
        printNumericTable(res.get(pca.eigenvalues), "Eigenvalues:")
        printNumericTable(res.get(pca.eigenvectors), "Eigenvectors:")
