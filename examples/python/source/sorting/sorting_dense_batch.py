# file: sorting_dense_batch.py
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
