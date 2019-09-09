# file: out_detect_uni_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-OUTLIER_DETECTION_UNIVARIATE_BATCH"></a>
## \example out_detect_uni_dense_batch.py

import os
import sys

from daal.algorithms import univariate_outlier_detection
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

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

    nFeatures = dataSource.getNumberOfColumns()

    algorithm = univariate_outlier_detection.Batch()

    algorithm.input.set(univariate_outlier_detection.data, dataSource.getNumericTable())

    # Compute outliers and get the computed results
    res = algorithm.compute()

    printNumericTable(dataSource.getNumericTable(), "Input data")
    printNumericTable(res.get(univariate_outlier_detection.weights), "Outlier detection result (univariate)")
