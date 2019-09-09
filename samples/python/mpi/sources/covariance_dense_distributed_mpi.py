# file: covariance_dense_distributed_mpi.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation
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
# !    Python sample of dense variance-covariance matrix computation in the
# !    distributed processing mode
# !
# !*******************************************************************************

## <a name="DAAL-SAMPLE-PY-COVARIANCE_DENSE_DISTRIBUTED"></a>
## \example covariance_dense_distributed_mpi.py

import os
import sys
from os.path import join as jp

from daal import step1Local, step2Master
from daal.algorithms import covariance
from daal.data_management import DataSourceIface, FileDataSource, OutputDataArchive, InputDataArchive

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

from mpi4py import MPI

# Input data set parameters
nBlocks = 4

MPI_ROOT = 0
DAAL_PREFIX = jp('data', 'distributed')
datasetFileNames = [
    jp(DAAL_PREFIX, 'covcormoments_dense_1.csv'),
    jp(DAAL_PREFIX, 'covcormoments_dense_2.csv'),
    jp(DAAL_PREFIX, 'covcormoments_dense_3.csv'),
    jp(DAAL_PREFIX, 'covcormoments_dense_4.csv')
]

if __name__ == '__main__':

    comm_size = MPI.COMM_WORLD
    rankId = comm_size.Get_rank()

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileNames[rankId],
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm to compute a variance-covariance matrix on local nodes
    localAlgorithm = covariance.Distributed(step1Local)

    # Set the input data set to the algorithm
    localAlgorithm.input.set(covariance.data, dataSource.getNumericTable())

    # Compute a variance-covariance matrix
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()

    pres.serialize(dataArch)
    perNodeArchLength = dataArch.getSizeOfArchive()

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    data = comm_size.gather(nodeResults, MPI_ROOT)

    if rankId == MPI_ROOT:

        # Create an algorithm to compute a variance-covariance matrix on the master node
        masterAlgorithm = covariance.Distributed(step2Master)

        for i in range(nBlocks):

            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(data[i])

            dataForStep2FromStep1 = covariance.PartialResult()

            dataForStep2FromStep1.deserialize(dataArch)

            # Set local partial results as input for the master-node algorithm
            masterAlgorithm.input.add(covariance.partialResults, dataForStep2FromStep1)

        # Merge and finalizeCompute a dense variance-covariance matrix on the master node */
        masterAlgorithm.compute()
        result = masterAlgorithm.finalizeCompute()

        # Print the results
        printNumericTable(result.get(covariance.covariance), "Covariance matrix:")
        printNumericTable(result.get(covariance.mean),       "Mean vector:")
