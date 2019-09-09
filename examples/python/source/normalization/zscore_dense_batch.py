# file: zscore_dense_batch.py
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
# !    Python example of Z-score normalization algorithm.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-ZSCORE_BATCH"></a>
## \example zscore_dense_batch.py
#

import os
import sys

import daal.algorithms.normalization.zscore as zscore
from daal.data_management import DataSourceIface, FileDataSource

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Input data set parameters
datasetName = os.path.join('..', 'data', 'batch', 'normalization.csv')

if __name__ == "__main__":

    # Retrieve the input data
    dataSource = FileDataSource(datasetName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)
    dataSource.loadDataBlock()

    data = dataSource.getNumericTable()

    # Create an algorithm
    algorithm = zscore.Batch(method=zscore.sumDense)

    # Set an input object for the algorithm
    algorithm.input.set(zscore.data, data)

    # Compute Z-score normalization function
    res = algorithm.compute()

    printNumericTable(data, "First 10 rows of the input data:", 10)
    printNumericTable(res.get(zscore.normalizedData), "First 10 rows of the z-score normalization result:", 10)
