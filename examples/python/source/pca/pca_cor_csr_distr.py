# file: pca_cor_csr_distr.py
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

## <a name="DAAL-EXAMPLE-PY-PCA_CORRELATION_CSR_DISTRIBUTED"></a>
## \example pca_cor_csr_distr.py

import os
import sys

import numpy as np

from daal import step1Local, step2Master
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

    # Create an algorithm for principal component analysis using the correlation method on the master node
    masterAlgorithm = pca.Distributed(step2Master,fptype=np.float64)

    for i in range(nBlocks):
        dataTable = createSparseTable(datasetFileNames[i])

        # Create algorithm objects to compute a variance-covariance matrix in the distributed processing mode using the default method
        localAlgorithm = pca.Distributed(step1Local,fptype=np.float64)

        # Create an algorithm for principal component analysis using the correlation method on the local node
        localAlgorithm.parameter.covariance = covariance.Distributed(step1Local, fptype=np.float64, method=covariance.fastCSR)

        # Set input objects for the algorithm
        localAlgorithm.input.setDataset(pca.data, dataTable)

        # Compute partial estimates on local nodes
        # Set local partial results as input for the master-node algorithm
        masterAlgorithm.input.add(pca.partialResults, localAlgorithm.compute())

    # Use covariance algorithm for sparse data inside the PCA algorithm
    masterAlgorithm.parameter.covariance = covariance.Distributed(step2Master, fptype=np.float64, method=covariance.fastCSR)

    # Merge and finalize PCA decomposition on the master node
    masterAlgorithm.compute()

    result = masterAlgorithm.finalizeCompute()

    # Print the results
    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
