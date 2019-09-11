# file: pca_cor_csr_online.py
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

## <a name="DAAL-EXAMPLE-PY-PCA_CORRELATION_CSR_ONLINE"></a>
## \example pca_cor_csr_online.py

import os
import sys

import numpy as np

from daal.algorithms import covariance
from daal.algorithms import pca

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4
datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_csr_4.csv')
]

if __name__ == "__main__":

    # Create an algorithm for principal component analysis using the correlation method
    algorithm = pca.Online(fptype=np.float64)

    # Use covariance algorithm for sparse data inside the PCA algorithm
    algorithm.parameter.covariance = covariance.Online(fptype=np.float64,method=covariance.fastCSR)

    for i in range(nBlocks):
        # Read data from a file and create a numeric table to store input data
        dataTable = createSparseTable(datasetFileNames[i])

        # Set input objects for the algorithm
        algorithm.input.setDataset(pca.data, dataTable)

        # Update PCA decomposition
        algorithm.compute()

    # Finalize computations
    result = algorithm.finalizeCompute()

    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
