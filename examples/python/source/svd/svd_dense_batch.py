# file: svd_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-SVD_BATCH"></a>
## \example svd_dense_batch.py

import os
import sys

from daal.algorithms import svd
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

#  Input data set parameters
datasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'svd.csv')
nRows = 16000

if __name__ == "__main__":

    # Initialize FileDataSource<CSVFeatureManager> to retrieve input data from .csv file
    dataSource = FileDataSource(
        datasetFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from input file
    dataSource.loadDataBlock(nRows)

    # Create algorithm to compute SVD decomposition
    algorithm = svd.Batch()

    algorithm.input.set(svd.data, dataSource.getNumericTable())

    # Compute SVD decomposition
    res = algorithm.compute()

    # Print results
    printNumericTable(res.get(svd.singularValues),      "Singular values:")
    printNumericTable(res.get(svd.rightSingularMatrix), "Right orthogonal matrix V:")
    printNumericTable(res.get(svd.leftSingularMatrix),  "Left orthogonal matrix U:", 10)
