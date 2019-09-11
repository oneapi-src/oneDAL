# file: qr_dense_online.py
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

## <a name="DAAL-EXAMPLE-PY-QR_ONLINE"></a>
## \example qr_dense_online.py

import os
import sys

from daal.algorithms import qr
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nRowsInBlock = 4000
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'qr.csv')

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve input data from .csv file
    dataSource = FileDataSource(
        dataFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create algorithm to compute QR decomposition
    algorithm = qr.Online()

    while dataSource.loadDataBlock(nRowsInBlock) == nRowsInBlock:
        # Set input arguments of the algorithm
        algorithm.input.set(qr.data, dataSource.getNumericTable())

        # Compute partial QR decompisition estimates
        algorithm.compute()

    # Finalize online result and get computed QR decomposition
    res = algorithm.finalizeCompute()

    # Print values
    printNumericTable(res.get(qr.matrixQ), "Orthogonal matrix Q:", 10)
    printNumericTable(res.get(qr.matrixR), "Triangular matrix R:")
