# file: spat_ave_pool2d_layer_dense_batch.py
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
# !    Python example of neural network forward and backward two-dimensional spatial pyramid average pooling layers usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SPAT_AVE_POOL2D_LAYER_DENSE_BATCH"></a>
## \example spat_ave_pool2d_layer_dense_batch.py
#

import os
import sys

import numpy as np

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, printNumericTable

nDim = 4
dims = [2, 3, 2, 4]
dataArray = np.array([[[[2, 4, 6, 8],
                        [10, 12, 14, 16]],
                       [[18, 20, 22, 24],
                        [26, 28, 30, 32]],
                       [[34, 36, 38, 40],
                        [42, 44, 46, 48]]],
                                          [[[-2,  -4,  -6,  -8],
                                            [-10, -12, -14, -16]],
                                           [[-18, -20, -22, -24],
                                            [-26, -28, -30, -32]],
                                           [[-34, -36, -38, -40],
                                            [-42, -44, -46, -48]]]],
                     dtype=np.float64)

if __name__ == "__main__":
    data = HomogenTensor(dataArray)

    # Read datasetFileName from a file and create a tensor to store input data
    printTensor(data, "Forward two-dimensional spatial pyramid average pooling layer input (first 10 rows):", 10)

    # Create an algorithm to compute forward two-dimensional maximum pooling layer results using default method
    forwardLayer = layers.spatial_average_pooling2d.forward.Batch(2, nDim)
    forwardLayer.input.setInput(layers.forward.data, data)

    # Compute forward two-dimensional spatial pyramid average pooling layer results and return them
    # Result class from layers.spatial_average_pooling2d.forward
    forwardResult = forwardLayer.compute()

    printTensor(forwardResult.getResult(layers.forward.value),
                "Forward two-dimensional spatial pyramid average pooling layer result (first 5 rows):",
                5)
    printNumericTable(forwardResult.getLayerData(layers.spatial_average_pooling2d.auxInputDimensions),
                      "Forward two-dimensional spatial pyramid average pooling layer input dimensions:")

    # Create an algorithm to compute backward two-dimensional spatial pyramid average pooling layer results using default method
    backwardLayer = layers.spatial_average_pooling2d.backward.Batch(2, nDim)
    backwardLayer.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward two-dimensional spatial pyramid average pooling layer results
    # Result class from layers.spatial_average_pooling2d.backward
    backwardResult = backwardLayer.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Backward two-dimensional spatial pyramid average pooling layer result (first 10 rows):",
                10)
