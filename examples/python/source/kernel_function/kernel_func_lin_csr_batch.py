# file: kernel_func_lin_csr_batch.py
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

#
# !  Content:
# !    Python example of computing a linear kernel function in the batch processing mode
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-KERNEL_FUNCTION_LINEAR_CSR_BATCH"></a>
## \example kernel_func_lin_csr_batch.py
#
import os
import sys

from daal.algorithms import kernel_function

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

data_dir = os.path.join('..', 'data', 'batch')
# Input data set parameters
leftDatasetFileName = os.path.join(data_dir, 'kernel_function_csr.csv')
rightDatasetFileName = os.path.join(data_dir, 'kernel_function_csr.csv')

# Kernel algorithm parameters
k = 1.0  # Linear kernel coefficient in the k(X,Y) + b model
b = 0.0  # Linear kernel coefficient in the k(X,Y) + b model

if __name__ == "__main__":

    # Read datasetFileName from a file and create a numeric tables to store input data
    leftData = createSparseTable(leftDatasetFileName)
    rightData = createSparseTable(rightDatasetFileName)

    # Create algorithm objects for the kernel algorithm using the default method
    algorithm = kernel_function.linear.Batch(method=kernel_function.linear.fastCSR)

    # Set the kernel algorithm parameter
    algorithm.parameter.k = k
    algorithm.parameter.b = b
    algorithm.parameter.computationMode = kernel_function.matrixMatrix

    # Set an input data table for the algorithm
    algorithm.input.set(kernel_function.X, leftData)
    algorithm.input.set(kernel_function.Y, rightData)

    # Compute the linear kernel function
    # (Result class from daal.algorithms.kernel_function)
    result = algorithm.compute()

    # Print the results
    printNumericTable(result.get(kernel_function.values), "Values")
