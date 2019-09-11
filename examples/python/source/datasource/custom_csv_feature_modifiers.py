# file: custom_csv_feature_modifiers.py
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

# !  Content:
# !    Python example of modifiers usage with file data source
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-DATASOURCE_CUSTOM_CSV_FEATURE_MODIFIERS">
## \example custom_csv_feature_modifiers.py
#

from daal.data_management import FileDataSource, CsvDataSourceOptions, modifiers
from daal.data_management.modifiers.csv import FeatureModifier

import os,  sys
utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# User-defined feature modifier that computes a square for every feature
class MySquaringModifier(FeatureModifier):
    def apply(self, tokens):
        return [[float(x)*float(x) for x in t] for t in tokens]


# User-defined feature modifier that selects max element among all features
class MyMaxFeatureModifier(FeatureModifier):
    def __init__(self):
        super(MyMaxFeatureModifier, self).__init__(1,4)

    # This method is called for every row in CSV file
    def apply(self, tokens):
            return [[float(max(t))] for t in tokens]


if __name__ == "__main__":
    # Path to the CSV to be read
    csvFileName = "../data/batch/mixed_text_and_numbers.csv"

    # Define options for CSV data source
    csvOptions = CsvDataSourceOptions(CsvDataSourceOptions.allocateNumericTable | CsvDataSourceOptions.createDictionaryFromContext | CsvDataSourceOptions.parseHeader)

    # Define CSV file data source
    ds = FileDataSource(csvFileName, csvOptions)

    # Configure format of output numeric table by applying modifiers.
    # Output numeric table will have the following format:
    # | Numeric1 | Numeric2 ^ 2 | Numeric5 ^ 2 | max(Numeric0, Numeric5) |
    fm = ds.getFeatureManager()
    fm.addModifier(["Numeric1"], modifiers.csv.continuous())
    fm.addModifier(["Numeric2", "Numeric5"], MySquaringModifier())
    fm.addModifier(["Numeric0", "Numeric5"], MyMaxFeatureModifier())

    # Load and parse CSV file
    ds.loadDataBlock()
    printNumericTable(ds.getNumericTable(), "Loaded numeric table:")
