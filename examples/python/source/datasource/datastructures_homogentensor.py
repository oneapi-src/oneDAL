# file: datastructures_homogentensor.py
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
# !    Python example of using homogeneous tensor data structures
# !*****************************************************************************

#
## <a name ="DAAL-EXAMPLE-PY-DATASTRUCTURES_HOMOGENTENSOR"> </a>
## \example datastructures_homogentensor.py
#
from __future__ import print_function

import numpy as np

from daal.data_management import HomogenTensor, SubtensorDescriptor, readWrite

if __name__ == "__main__":

    data = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]],[[21,22,23],[24,25,26],[27,28,29]]],
                    dtype=np.float64)

    print("Initial data:")
    for i in data.flatten():
        print("{0:5.1f}".format(i), end=' ')
    print()

    hc = HomogenTensor(data)

    subtensor = SubtensorDescriptor()
    fDimN = 2
    fDims = [0, 1]
    hc.getSubtensor(fDims, 1, 2, readWrite, subtensor)

    d = subtensor.getNumberOfDims()
    print("Subtensor dimensions: {}".format(d))
    n = subtensor.getSize()
    print("Subtensor size:       {}".format(n))
    p = subtensor.getArray()
    print("Subtensor data:")
    for i in p:
        print("{0:5.1f}".format(i), end=' ')
    print()

    p[0] = -1

    hc.releaseSubtensor(subtensor)

    print("Data after modification:")
    for i in data.flatten():
        print("{0:5.1f}".format(i), end=' ')
    print()
