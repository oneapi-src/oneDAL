# file: pca_svd_dense_distr.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
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
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-PCA_SVD_DENSE_DISTRIBUTED"></a>
## \example pca_svd_dense_distr.py

import os
import sys

import daal
from daal.algorithms import pca
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4
nVectorsInBlock = 250

dataFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'pca_normalized_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'pca_normalized_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'pca_normalized_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'pca_normalized_4.csv')
]

if __name__ == "__main__":

    # Create an algorithm for principal component analysis using the SVD method on the master node
    masterAlgorithm = pca.Distributed(step=daal.step2Master, method=pca.svdDense)

    for i in range(nBlocks):
        # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
        dataSource = FileDataSource(
            dataFileNames[i], DataSourceIface.doAllocateNumericTable,
            DataSourceIface.doDictionaryFromContext
        )

        # Retrieve the input data
        dataSource.loadDataBlock(nVectorsInBlock)

        # Create an algorithm for principal component analysis using the SVD method on the local node
        localAlgorithm = pca.Distributed(step=daal.step1Local, method=pca.svdDense)

        # Set the input data to the algorithm
        localAlgorithm.input.setDataset(pca.data, dataSource.getNumericTable())

        # Compute PCA decomposition
        # Set local partial results as input for the master-node algorithm
        masterAlgorithm.input.add(pca.partialResults, localAlgorithm.compute())

    # Merge and finalize PCA decomposition on the master node
    masterAlgorithm.compute()
    result = masterAlgorithm.finalizeCompute()

    # Print the results
    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
