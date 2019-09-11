# file: quantiles_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-QUANTILES_BATCH"></a>
## \example quantiles_dense_batch.py

import os
import sys

from daal.algorithms import quantiles
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
dataFileName = os.path.join(DAAL_PREFIX, 'batch', 'quantiles.csv')

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve input data from .csv file
    dataSource = FileDataSource(
        dataFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from input file
    dataSource.loadDataBlock()

    # Create algorithm to compute quantiles in batch mode
    algorithm = quantiles.Batch()

    # Set input arguments of the algorithm
    algorithm.input.set(quantiles.data, dataSource.getNumericTable())

    # Get computed quantiles
    res = algorithm.compute()

    # Print result
    printNumericTable(res.get(quantiles.quantiles), "Quantiles")
