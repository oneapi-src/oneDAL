# file: pca_svd_dense_online.py
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

## <a name="DAAL-EXAMPLE-PY-PCA_SVD_DENSE_ONLINE"></a>
## \example pca_svd_dense_online.py

import os
import sys

from daal.algorithms import pca
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nVectorsInBlock = 250
dataFileName = os.path.join(DAAL_PREFIX, 'online', 'pca_normalized.csv')

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        dataFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create an algorithm for principal component analysis using the SVD method
    algorithm = pca.Online(method=pca.svdDense)

    while(dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock):
        # Set the input data to the algorithm
        algorithm.input.setDataset(pca.data, dataSource.getNumericTable())

        # Update PCA decomposition
        algorithm.compute()

    # Finalize computations
    result = algorithm.finalizeCompute()

    # Print the results
    printNumericTable(result.get(pca.eigenvalues), "Eigenvalues:")
    printNumericTable(result.get(pca.eigenvectors), "Eigenvectors:")
