# file: em_gmm_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-EM_GMM_BATCH"></a>
## \example em_gmm_dense_batch.py

import os
import sys

from daal.algorithms import em_gmm
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'em_gmm.csv')
nComponents = 2

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    nFeatures = dataSource.getNumberOfColumns()

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Create algorithm objects to initialize the EM algorithm for the GMM
    # computing the number of components using the default method
    initAlgorithm = em_gmm.init.Batch(nComponents)

    # Set an input data table for the initialization algorithm
    initAlgorithm.input.set(em_gmm.init.data, dataSource.getNumericTable())

    # Compute initial values for the EM algorithm for the GMM with the default parameters
    resultInit = initAlgorithm.compute()

    # Create algorithm objects for the EM algorithm for the GMM computing the number of components using the default method
    algorithm = em_gmm.Batch(nComponents)

    # Set an input data table for the algorithm
    algorithm.input.setTable(em_gmm.data, dataSource.getNumericTable())
    algorithm.input.setValues(em_gmm.inputValues, resultInit)

    # Compute the results of the EM algorithm for the GMM with the default parameters
    result = algorithm.compute()

    # Print the results
    printNumericTable(result.getResult(em_gmm.weights), "Weights")
    printNumericTable(result.getResult(em_gmm.means), "Means")
    for i in range(nComponents):
        printNumericTable(result.getCovariances(em_gmm.covariances, i), "Covariance")
