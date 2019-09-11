# file: svd_dense_online.py
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

## <a name="DAAL-EXAMPLE-PY-SVD_ONLINE"></a>
## \example svd_dense_online.py

import os
import sys
import numpy as np

from daal.algorithms import svd
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

#  Input data set parameters
nRowsInBlock = 4000
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'svd.csv')

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve input data from .csv file
    dataSource = FileDataSource(
        dataFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create algorithm object to compute SVD decomposition in online mode
    algorithm = svd.Online(fptype=np.float64)

    while dataSource.loadDataBlock(nRowsInBlock):
        # Set input arguments of the algorithm
        algorithm.input.set(svd.data, dataSource.getNumericTable())

        # Compute partial SVD decomposition estimates
        algorithm.compute()

    # Finalize online result and get computed SVD decomposition
    res = algorithm.finalizeCompute()

    # Print results
    printNumericTable(res.get(svd.singularValues),      "Singular values:")
    printNumericTable(res.get(svd.rightSingularMatrix), "Right orthogonal matrix V:")
    printNumericTable(res.get(svd.leftSingularMatrix),  "Left orthogonal matrix U:", 10)
