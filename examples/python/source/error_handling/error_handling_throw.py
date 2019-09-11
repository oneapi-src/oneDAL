# file: error_handling_throw.py
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

## <a name="DAAL-EXAMPLE-PY-ERROR_HANDLING_THROW"></a>
## \example error_handling_throw.py

import os

from daal.data_management import FileDataSource

wrongDatasetFileName = os.path.join('..', 'data', 'batch', 'wrong.csv')

if __name__ == "__main__":

    try:
        # Initialize FileDataSource to retrieve the input data from a .csv file
        wrongDataSource = FileDataSource(wrongDatasetFileName)
        # An exception should be generated

    except SystemError as e:
        # Retrieve the description of the generated exception.
        print("FileDataSource expected error: Error on file open\n")
