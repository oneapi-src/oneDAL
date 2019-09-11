# file: cov_dense_online.py
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

## <a name="DAAL-EXAMPLE-PY-COVARIANCE_DENSE_ONLINE"></a>
## \example cov_dense_online.py

import os
import sys

from daal.algorithms import covariance
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'covcormoments_dense.csv')
nObservations = 50

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve input data from .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create algorithm objects for covariance matrix computing in online mode using default method
    algorithm = covariance.Online()

    while (dataSource.loadDataBlock(nObservations) == nObservations):
        # Set input arguments of the algorithm
        algorithm.input.set(covariance.data, dataSource.getNumericTable())

        # Compute partial covariance estimates
        algorithm.compute()

    # Finalize online result and get computed covariance
    res = algorithm.finalizeCompute()

    printNumericTable(res.get(covariance.covariance), "Covariance matrix:")
    printNumericTable(res.get(covariance.mean),       "Mean vector:")
