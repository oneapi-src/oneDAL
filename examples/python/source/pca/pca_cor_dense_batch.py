# file: pca_cor_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-PCA_CORRELATION_DENSE_BATCH"></a>
## \example pca_cor_dense_batch.py

import os
import sys
import numpy as np

from daal.algorithms import pca
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'pca_normalized.csv')
nVectors = 1000

if __name__ == "__main__":
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file

    dataSource = FileDataSource(
        dataFileName, DataSourceIface.doAllocateNumericTable, DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock(nVectors)

    # Create an algorithm for principal component analysis using the correlation method
    algorithm = pca.Batch(fptype=np.float64)

    # Set the algorithm input data
    algorithm.input.setDataset(pca.data, dataSource.getNumericTable())
    algorithm.parameter.resultsToCompute = pca.mean | pca.variance | pca.eigenvalue;
    algorithm.parameter.isDeterministic = True;

    # Compute results of the PCA algorithm
    result = algorithm.compute()

    # Print the results
    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
    printNumericTable(result.get(pca.means), "Means:")
    printNumericTable(result.get(pca.variances), "Variances:")
