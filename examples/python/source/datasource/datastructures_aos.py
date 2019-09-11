# file: datastructures_aos.py
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

## <a name="DAAL-EXAMPLE-PY-DATASTRUCTURES_AOS"></a>
## @example datastructures_aos.py

import os
import sys

import numpy as np

from daal.data_management import features, AOSNumericTable, BlockDescriptor, readOnly, readWrite

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printArray


if __name__ == "__main__":

    print("Array of structures (AOS) numeric table example\n")

    points = np.array([(0.5, -1.3, 1, 100.1),
                       (2.5, -3.3, 2, 200.2),
                       (4.5, -5.3, 2, 350.3),
                       (6.5, -7.3, 0, 470.4),
                       (8.5, -9.3, 1, 270.5)],
                      dtype=[('x','f4'), ('y','f4'), ('categ','i4'), ('value','f8')])

    nObservations = len(points)
    nFeatures = len(points[0])

    # Construct AOS numericTable for a data array with nFeatures fields and nObservations elements
    # Dictionary will be initialized with type information from ndarray
    dataTable = AOSNumericTable(points)

    #  Get the dictionary and update it with additional information about data
    dict = dataTable.getDictionary()

    #  Add a feature type to the dictionary
    dict[0].featureType = features.DAAL_CONTINUOUS
    dict[1].featureType = features.DAAL_CONTINUOUS
    dict[2].featureType = features.DAAL_CATEGORICAL
    dict[3].featureType = features.DAAL_CONTINUOUS

    #  Set the number of categories for a categorical feature
    dict[2].categoryNumber = 3

    #  Read a block of rows
    firstReadRow = 0
    doubleBlock = BlockDescriptor()
    dataTable.getBlockOfRows(firstReadRow, nObservations, readWrite, doubleBlock)
    printArray(
        doubleBlock.getArray(), nFeatures, doubleBlock.getNumberOfRows(),
        doubleBlock.getNumberOfColumns(),"Print AOS data structures as double:"
    )
    dataTable.releaseBlockOfRows(doubleBlock)

    #  Read a feature (column)
    readFeatureIdx = 2

    intBlock = BlockDescriptor(ntype=np.intc)
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, intBlock)
    printArray(
        intBlock.getArray(), 1, intBlock.getNumberOfRows(), intBlock.getNumberOfColumns(),
        "Print the third feature of AOS:", flt64=False
    )
    dataTable.releaseBlockOfColumnValues(intBlock)
