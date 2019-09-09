# file: cov_dense_distr.py
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

## <a name="DAAL-EXAMPLE-PY-COVARIANCE_DENSE_DISTRIBUTED"></a>
## \example cov_dense_distr.py

import os
import sys

from daal import step1Local, step2Master
from daal.algorithms import covariance
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_dense_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_dense_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_dense_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'covcormoments_dense_4.csv')
]

partialResult = [0] * nBlocks
result = None


def computestep1Local(block):
    global partialResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileNames[block], DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Create algorithm objects to compute a dense variance-covariance matrix in the distributed processing mode using the default method
    algorithm = covariance.Distributed(step1Local)

    # Set input objects for the algorithm
    algorithm.input.set(covariance.data, dataSource.getNumericTable())

    # Compute partial estimates on local nodes
    partialResult[block] = algorithm.compute()  # Get the computed partial estimates


def computeOnMasterNode():
    global result

    # Create algorithm objects to compute a dense variance-covariance matrix in the distributed processing mode using the default method
    algorithm = covariance.Distributed(step2Master)

    # Set input objects for the algorithm
    for i in range(nBlocks):
        algorithm.input.add(covariance.partialResults, partialResult[i])

    # Compute a partial estimate on the master node from the partial estimates on local nodes
    algorithm.compute()

    # Finalize the result in the distributed processing mode
    result = algorithm.finalizeCompute() # Get the computed dense variance-covariance matrix

if __name__ == "__main__":

    for i in range(nBlocks):
        computestep1Local(i)

    computeOnMasterNode()

    printNumericTable(result.get(covariance.covariance), "Covariance matrix:")
    printNumericTable(result.get(covariance.mean),       "Mean vector:")
