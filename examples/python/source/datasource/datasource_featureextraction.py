# file: datasource_featureextraction.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#
# !  Content:
# !    Python example for using of data source feature extraction
# !*****************************************************************************

#
## <a name = "DAAL-EXAMPLE-PY-DATASOURCE_FEATUREEXTRACTION"></a>
## \example datasource_featureextraction.py
#
import os
import sys

from daal.data_management import FileDataSource, DataSourceIface, ColumnFilter, OneHotEncoder

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


# Input data set parameters
datasetFileName = "../data/batch/kmeans_dense.csv"

if __name__ == "__main__":

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileName, DataSourceIface.doAllocateNumericTable)

    # Create data source dictionary from loading of the first .csv file
    dataSource.createDictionaryFromContext()

    # Filter in 3 chosen columns from a .csv file
    validList = [1, 2, 5]

    colFilter = ColumnFilter()
    filterList = colFilter.list(validList)
    dataSource.getFeatureManager().addModifier(filterList)

    # Consider column with index 1 as categorical and convert it into 3 binary categorical features
    dataSource.getFeatureManager().addModifier(OneHotEncoder(1, 3))

    # Load data from .csv file
    dataSource.loadDataBlock()

    # Print result
    table = dataSource.getNumericTable()
    printNumericTable(table, "Loaded data", 4, 20)
