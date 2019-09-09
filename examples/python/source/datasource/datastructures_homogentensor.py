# file: datastructures_homogentensor.py
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
