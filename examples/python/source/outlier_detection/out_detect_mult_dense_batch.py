# file: out_detect_mult_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-OUTLIER_DETECTION_MULTIVARIATE_DENSE_BATCH"></a>
## \example out_detect_mult_dense_batch.py

import os
import sys

from daal.algorithms import multivariate_outlier_detection
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'outlierdetection.csv')

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    dataSource = FileDataSource(
        datasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from the input file
    dataSource.loadDataBlock()

    # Create an algorithm to detect outliers using the default method
    algorithm = multivariate_outlier_detection.Batch()

    algorithm.input.set(multivariate_outlier_detection.data, dataSource.getNumericTable())

    # Compute outliers and get the computed results
    res = algorithm.compute()

    printNumericTables(
        dataSource.getNumericTable(),
        res.get(multivariate_outlier_detection.weights),
        "Input data", "Weights",
        "Outlier detection result (Default method)"
    )
