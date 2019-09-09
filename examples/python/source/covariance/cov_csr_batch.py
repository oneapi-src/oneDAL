# file: cov_csr_batch.py
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

## <a name="DAAL-EXAMPLE-PY-COVARIANCE_CSR_BATCH"></a>
## \example cov_csr_batch.py

import os
import sys

from daal.algorithms import covariance

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

#  Input matrix is stored in one-based sparse row storage format
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'covcormoments_csr.csv')

if __name__ == "__main__":

    # Read datasetFileName from file and create numeric table for storing input data
    dataTable = createSparseTable(datasetFileName)

    # Create algorithm to compute covariance matrix using default method
    algorithm = covariance.Batch()
    algorithm.input.set(covariance.data, dataTable)

    # Get computed covariance
    res = algorithm.compute()

    printNumericTable(res.get(covariance.covariance), "Covariance matrix (upper left square 10*10) :", 10, 10)
    printNumericTable(res.get(covariance.mean),       "Mean vector:", 1, 10)
