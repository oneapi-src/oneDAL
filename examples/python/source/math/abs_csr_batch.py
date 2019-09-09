# file: abs_csr_batch.py
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
# !    Python example of abs algorithm.
# !
# !*****************************************************************************

# *
## <a name="DAAL-EXAMPLE-PY-ABS_CSR_BATCH"></a>
## \example abs_csr_batch.py
#

import os
import sys

import daal.algorithms.math.abs
from daal.algorithms import math

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
    algorithm = math.abs.Batch(method=math.abs.fastCSR)

    # Set an input object for the algorithm
    algorithm.input.set(math.abs.data, dataTable)

    # Compute Abs function
    res = algorithm.compute()

    # Print the results of the algorithm
    printNumericTable(res.get(math.abs.value), "Abs result (first 5 rows):", 5)
