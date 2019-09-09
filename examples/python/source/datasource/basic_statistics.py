# file: basic_statistics.py
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
# !    Python example for using of basic statistics
# !*****************************************************************************

#
## <a name = "DAAL-EXAMPLE-PY-BASIC_STATISTICS"></a>
## \example basic_statistics.py
#

import os
import sys
import numpy as np

from daal.data_management import HomogenNumericTable, NumericTableIface, FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


if __name__ == "__main__":

    print("Basic statistics example\n")

    # Input data set parameters
    datasetFileName = "../data/batch/basic_statistics.csv"
    data = np.array([(7.0, 3.0, 6.0, 2.0),
                     (1.0, 3.0, 0.0, 2.0),
                     (9.0, 2.0, 6.0, 2.0),
                     (3.0, 4.0, 7.0, 2.0),])

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName, DataSourceIface.doAllocateNumericTable)

    dataSource.createDictionaryFromContext()
    dataSource.loadDataBlock()
    table = dataSource.getNumericTable()

    # Get basic statistics from the table. They were calculated inside DataSource for each column.
    min = table.basicStatistics.get(NumericTableIface.minimum)
    max = table.basicStatistics.get(NumericTableIface.maximum)
    sum = table.basicStatistics.get(NumericTableIface.sum)
    sumSquares = table.basicStatistics.get(NumericTableIface.sumSquares)

    # Print calculated basic statistics
    printNumericTable(table,      "Basic statistics of table:")
    printNumericTable(min,        "Minimum:")
    printNumericTable(max,        "Maximum:")
    printNumericTable(sum,        "Sum:")
    printNumericTable(sumSquares, "SumSquares:")

    # Create NumericTable with the same data. But in this case basic statistics are not calculated.
    dataTable = HomogenNumericTable(data)

    # Set basic statistics in the new NumericTable
    dataTable.basicStatistics.set(NumericTableIface.minimum, min);
    dataTable.basicStatistics.set(NumericTableIface.maximum, max);
    dataTable.basicStatistics.set(NumericTableIface.sum, sum);
    dataTable.basicStatistics.set(NumericTableIface.sumSquares, sumSquares);

    # Print basic statistics those were set
    printNumericTable(dataTable,                                                   "New table:")
    printNumericTable(dataTable.basicStatistics.get(NumericTableIface.minimum),    "Minimum:")
    printNumericTable(dataTable.basicStatistics.get(NumericTableIface.maximum),    "Maximum:")
    printNumericTable(dataTable.basicStatistics.get(NumericTableIface.sum),        "Sum:")
    printNumericTable(dataTable.basicStatistics.get(NumericTableIface.sumSquares), "SumSquares:")
