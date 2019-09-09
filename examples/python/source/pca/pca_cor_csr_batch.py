# file: pca_cor_csr_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

## <a name="DAAL-EXAMPLE-PY-PCA_CORRELATION_CSR_BATCH"></a>
## \example pca_cor_csr_batch.py

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
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'covcormoments_csr.csv')

if __name__ == "__main__":

    # Read data from a file and create a numeric table to store input data
    dataTable = createSparseTable(dataFileName)

    # Create an algorithm for principal component analysis using the correlation method
    algorithm = pca.Batch(fptype=np.float64, method=pca.correlationDense)

    # Use covariance algorithm for sparse data inside the PCA algorithm
    algorithm.parameter.covariance = covariance.Batch(fptype=np.float64, method=covariance.fastCSR)

    # Set the algorithm input data
    algorithm.input.setDataset(pca.data, dataTable)
    algorithm.parameter.resultsToCompute = pca.mean | pca.variance | pca.eigenvalue;
    algorithm.parameter.isDeterministic = True;
    # Compute results of the PCA algorithm
    result = algorithm.compute()

    # Print the results
    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
    printNumericTable(result.get(pca.means), "Means:")
    printNumericTable(result.get(pca.variances), "Variances:")
