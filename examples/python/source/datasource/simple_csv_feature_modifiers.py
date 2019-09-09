# file: simple_csv_feature_modifiers.py
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

# !  Content:
# !    Python example of modifiers usage with file data source
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-DATASOURCE_SIMPLE_CSV_FEATURE_MODIFIERS">
## \example simple_csv_feature_modifiers.py
#

from daal.data_management import FileDataSource, CsvDataSourceOptions, modifiers, features
from daal.data_management.modifiers import csv

import os,  sys
utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Path to the CSV to be read
csvFileName = "../data/batch/mixed_text_and_numbers.csv"

# Define options for CSV data source
csvOptions = CsvDataSourceOptions(CsvDataSourceOptions.allocateNumericTable |\
                                  CsvDataSourceOptions.createDictionaryFromContext |\
                                  CsvDataSourceOptions.parseHeader)

# Read CSV using default data source behavior
def readDefault():
    ds = FileDataSource(csvFileName, csvOptions)
    # By default all numeric columns will be parsed as continuous
    # features and other columns as categorical
    ds.loadDataBlock()
    printNumericTable(ds.getNumericTable(), "readDefault function result:")


# Read CSV and do basic filtering using columns indices
def readOnlySpecifiedColumnIndices():
    ds = FileDataSource(csvFileName, csvOptions)
    # This means that columns with indices 0, 1, 5 will be included to the output numeric
    # table and other columns will be ignored. The first argument of method 'include' specifies
    # the set of columns and the second one specifies modifier. in this case we use predefined
    # automatic modifier that automatically decides how to parse column in the best way
    print(modifiers.csv.automatic())
    ds.getFeatureManager().addModifier([0,1,5], modifiers.csv.automatic())
    ds.loadDataBlock()
    printNumericTable(ds.getNumericTable(), "readOnlySpecifiedColumnIndices function result:")


# Read CSV and do basic filtering using columns names
def readOnlySpecifiedColumnNames():
    ds = FileDataSource(csvFileName, csvOptions)
    # The same as readOnlySpecifiedColumnIndices but uses column names instead of indices
    ds.getFeatureManager().addModifier(["Numeric1", "Categorical0"], modifiers.csv.automatic())
    ds.loadDataBlock()
    printNumericTable(ds.getNumericTable(), "readOnlySpecifiedColumnNames function result:")


# Read CSV using multiple modifiers
def readUsingMultipleModifiers():
    ds = FileDataSource(csvFileName, csvOptions)

    fm = ds.getFeatureManager()
    fm.addModifier(["Numeric1"], modifiers.csv.continuous())
    # let's mix position and names
    fm.addModifier([6, "Categorical1"], modifiers.csv.categorical())

    ds.loadDataBlock()
    printNumericTable(ds.getNumericTable(), "readUsingMultipleModifiers function result:")


if __name__ == "__main__":
    # Read CSV using default data source behavior
    readDefault()

    # Read CSV and do basic filtering using columns indices
    readOnlySpecifiedColumnIndices()

    # Read CSV and do basic filtering using columns names
    readOnlySpecifiedColumnNames()

    # Read CSV using multiple modifiers
    readUsingMultipleModifiers()
