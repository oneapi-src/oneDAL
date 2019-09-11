# file: sorting_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-SORTING_BATCH"></a>
## \example sorting_dense_batch.py

import os
import sys

from daal.algorithms import sorting
from daal.data_management import DataSourceIface, FileDataSource

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Input data set parameters
datasetFileName = os.path.join('..', 'data', 'batch', 'sorting.csv')

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Create algorithm objects to sort data using the default (radix) method
    algorithm = sorting.Batch()

    # Print the input observations matrix
    printNumericTable(dataSource.getNumericTable(), "Initial matrix of observations:")

    # Set input objects for the algorithm
    algorithm.input.set(sorting.data, dataSource.getNumericTable())

    # Sort data observations
    res = algorithm.compute()

    printNumericTable(res.get(sorting.sortedData), "Sorted matrix of observations:")
