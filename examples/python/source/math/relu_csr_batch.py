# file: relu_csr_batch.py
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

#
# !  Content:
# !    Python example of ReLU algorithm.
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-RELU_CSR_BATCH"></a>
## \example relu_csr_batch.py
#

import os
import sys

import daal.algorithms.math.relu as relu

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

# Input data set parameters
datasetName = os.path.join('..', 'data', 'batch', 'covcormoments_csr.csv')

if __name__ == "__main__":

    # Read datasetFileName from a file and create a numeric table to store input data
    dataTable = createSparseTable(datasetName)

    # Create an algorithm
    algorithm = relu.Batch(method=relu.fastCSR)

    # Set an input object for the algorithm
    algorithm.input.set(relu.data, dataTable)

    # Compute ReLU function
    res = algorithm.compute()

    # Print the results of the algorithm
    printNumericTable(res.get(relu.value), "ReLU result (first 5 rows):", 5)
