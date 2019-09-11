# file: spat_max_pool2d_layer_dense_batch.py
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
# !    Python example of neural network forward and backward two-dimensional spatial pyramid maximum pooling layers usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SPAT_MAX_POOL2D_LAYER_DENSE_BATCH"></a>
## \example spat_max_pool2d_layer_dense_batch.py
#

import os
import sys

import numpy as np

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import spatial_maximum_pooling2d
from daal.data_management import HomogenTensor

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor

nDim = 4
dims = [2, 3, 2, 4]
dataArray = np.array([[[[1,  2,  3,  4],
                        [5,  6,  7,  8]],
                       [[9, 10, 11, 12],
                        [13, 14, 15, 16]],
                       [[17, 18, 19, 20],
                        [21, 22, 23, 24]]],
                                         [[[-1,  -2,  -3,  -4],
                                           [-5,  -6,  -7,  -8]],
                                          [[-9, -10, -11, -12],
                                           [-13, -14, -15, -16]],
                                          [[-17, -18, -19, -20],
                                           [-21, -22, -23, -24]]]],
                     dtype=np.float64)

if __name__ == "__main__":
    data = HomogenTensor(dataArray)


    printTensor(data, "Forward two-dimensional spatial pyramid maximum pooling layer input (first 10 rows):", 10)

    # Create an algorithm to compute forward two-dimensional spatial pyramid maximum pooling layer results using default method
    forwardLayer = spatial_maximum_pooling2d.forward.Batch(2, nDim)
    forwardLayer.input.setInput(layers.forward.data, data)

    # Compute forward two-dimensional spatial pyramid maximum pooling layer results
    forwardLayer.compute()

    # Get the computed forward two-dimensional spatial pyramid maximum pooling layer results
    forwardResult = forwardLayer.getResult()

    printTensor(forwardResult.getResult(layers.forward.value), "Forward two-dimensional spatial pyramid maximum pooling layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.spatial_maximum_pooling2d.auxSelectedIndices),
                "Forward two-dimensional spatial pyramid maximum pooling layer selected indices (first 10 rows):", 10)

    # Create an algorithm to compute backward two-dimensional spatial pyramid maximum pooling layer results using default method
    backwardLayer = layers.spatial_maximum_pooling2d.backward.Batch(2, nDim)
    backwardLayer.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward two-dimensional spatial pyramid maximum pooling layer results
    backwardLayer.compute()

    # Get the computed backward two-dimensional spatial pyramid maximum pooling layer results
    backwardResult = backwardLayer.getResult()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Backward two-dimensional spatial pyramid maximum pooling layer result (first 10 rows):", 10)
