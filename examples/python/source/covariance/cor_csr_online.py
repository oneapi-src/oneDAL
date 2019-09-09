# file: cor_csr_online.py
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

## <a name="DAAL-EXAMPLE-PY-CORRELATION_CSR_ONLINE"></a>
## \example cor_csr_online.py

import os
import sys

from daal.algorithms import covariance

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
    os.path.join(DAAL_PREFIX, 'online', 'covcormoments_csr_4.csv'),
]

if __name__ == "__main__":

    # Create algorithm objects for correlation matrix computing in online mode using default method
    algorithm = covariance.Online()

    # Set the parameter to choose the type of the output matrix
    algorithm.parameter.outputMatrixType = covariance.correlationMatrix

    for i in range(nBlocks):
        dataTable = createSparseTable(datasetFileNames[i])

        # Set input arguments of the algorithm
        algorithm.input.set(covariance.data, dataTable)

        # Compute partial correlation estimates
        algorithm.compute()

    # Finalize online result and get computed correlation
    res = algorithm.finalizeCompute()

    printNumericTable(res.get(covariance.correlation), "Correlation matrix (upper left square 10*10) :", 10, 10)
    printNumericTable(res.get(covariance.mean),        "Mean vector:", 1, 10)
