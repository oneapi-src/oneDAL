# file: kernel_func_rbf_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-KERNEL_FUNCTION_RBF_DENSE_BATCH"></a>
## \example kernel_func_rbf_dense_batch.py

import os
import sys

from daal.algorithms import kernel_function
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
leftDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'kernel_function.csv')
rightDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'kernel_function.csv')

# Kernel algorithm parameters
sigma = 1.0

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    leftDataSource = FileDataSource(
        leftDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    rightDataSource = FileDataSource(
        rightDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    leftDataSource.loadDataBlock()
    rightDataSource.loadDataBlock()

    # Create algorithm objects for the kernel algorithm using the default method
    algorithm = kernel_function.rbf.Batch()

    # Set the kernel algorithm parameter
    algorithm.parameter.sigma = sigma
    algorithm.parameter.computationMode = kernel_function.matrixMatrix

    # Set an input data table for the algorithm
    algorithm.input.set(kernel_function.X, leftDataSource.getNumericTable())
    algorithm.input.set(kernel_function.Y, rightDataSource.getNumericTable())

    # Compute the RBF kernel and get the computed results
    result = algorithm.compute()

    # Print the results
    printNumericTable(result.get(kernel_function.values), "Values")
