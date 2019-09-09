# file: low_order_moms_csr_online.py
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

## <a name="DAAL-EXAMPLE-PY-LOW_ORDER_MOMENTS_CSR_ONLINE"></a>
## \example low_order_moms_csr_online.py

import os
import sys

from daal.algorithms import low_order_moments

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'online', 'covcormoments_csr_1.csv'),
    os.path.join(DAAL_PREFIX, 'online', 'covcormoments_csr_2.csv'),
    os.path.join(DAAL_PREFIX, 'online', 'covcormoments_csr_3.csv'),
    os.path.join(DAAL_PREFIX, 'online', 'covcormoments_csr_4.csv')
]


def printResults(res):
    printNumericTable(res.get(low_order_moments.minimum),              "Minimum:")
    printNumericTable(res.get(low_order_moments.maximum),              "Maximum:")
    printNumericTable(res.get(low_order_moments.sum),                  "Sum:")
    printNumericTable(res.get(low_order_moments.sumSquares),           "Sum of squares:")
    printNumericTable(res.get(low_order_moments.sumSquaresCentered),   "Sum of squared difference from the means:")
    printNumericTable(res.get(low_order_moments.mean),                 "Mean:")
    printNumericTable(res.get(low_order_moments.secondOrderRawMoment), "Second order raw moment:")
    printNumericTable(res.get(low_order_moments.variance),             "Variance:")
    printNumericTable(res.get(low_order_moments.standardDeviation),    "Standard deviation:")
    printNumericTable(res.get(low_order_moments.variation),            "Variation:")

if __name__ == "__main__":

    # Create algorithm objects for low order moments computing in online mode using default method
    algorithm = low_order_moments.Online()

    for i in range(nBlocks):
        dataTable = createSparseTable(datasetFileNames[i])

        # Set input arguments of the algorithm
        algorithm.input.set(low_order_moments.data, dataTable)

        # Compute partial low order moments estimates
        algorithm.compute()

    # Finalize online result and get computed low order moments
    res = algorithm.finalizeCompute()

    printResults(res)
