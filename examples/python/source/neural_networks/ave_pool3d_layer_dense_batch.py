# file: ave_pool3d_layer_dense_batch.py
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
# !    Python example of neural network forward and backward three-dimensional average pooling layers usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-AVERAGE_POOLING3D_LAYER_BATCH"></a>
## \example ave_pool3d_layer_dense_batch.py
#

import os
import sys

import numpy as np

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor3d, printNumericTable

nDim = 3
dims = [3, 2, 4]
dataArray = np.array([[[1,  2,  3,  4],
                       [5,  6,  7,  8]],
                      [[9, 10, 11, 12],
                       [13, 14, 15, 16]],
                      [[17, 18, 19, 20],
                       [21, 22, 23, 24]]],
                     dtype=np.float64)

if __name__ == "__main__":

    dataTensor = HomogenTensor(dataArray)

    printTensor3d(dataTensor, "Forward average pooling layer input:")

    # Create an algorithm to compute forward pooling layer results using average method
    forwardLayer = layers.average_pooling3d.forward.Batch(nDim)
    forwardLayer.input.setInput(layers.forward.data, dataTensor)

    # Compute forward pooling layer results
    # Result class from layers.average_pooling3d.forward
    forwardResult = forwardLayer.compute()

    printTensor3d(forwardResult.getResult(layers.forward.value), "Forward average pooling layer result:")
    printNumericTable(forwardResult.getLayerData(layers.average_pooling3d.auxInputDimensions),
                      "Forward pooling layer input dimensions:")

    # Create an algorithm to compute backward pooling layer results using average method
    backwardLayer = layers.average_pooling3d.backward.Batch(nDim)
    backwardLayer.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward pooling layer results
    # Result class from layers.average_pooling3d.backward
    backwardResult = backwardLayer.compute()

    printTensor3d(backwardResult.getResult(layers.backward.gradient), "Backward average pooling layer result:")
