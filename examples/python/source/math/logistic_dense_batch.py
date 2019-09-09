# file: logistic_dense_batch.py
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
# !    Python example of Logistic algorithm.
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LOGISTIC_BATCH"></a>
## \example logistic_dense_batch.py
#

import os
import sys

import daal.algorithms.math.logistic as logistic
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Input data set parameters
datasetName = os.path.join('..', 'data', 'batch', 'covcormoments_dense.csv')

if __name__ == "__main__":

    # Retrieve the input data
    dataSource = FileDataSource(datasetName,
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)
    dataSource.loadDataBlock()

    # Create an algorithm
    algorithm = logistic.Batch()

    # Set an input object for the algorithm
    algorithm.input.set(logistic.data, dataSource.getNumericTable())

    # Compute Logistic function
    res = algorithm.compute()

    # Print the results of the algorithm
    printNumericTable(res.get(logistic.value), "Logistic result (first 5 rows):", 5)
