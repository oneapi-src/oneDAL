# file: locallycon2d_layer_dense_batch.py
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
# !    Python example of forward and backward 2D locally connected layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LOCALLYCONNECTED2D_LAYER_BATCH"></a>
## \example locallycon2d_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor, TensorIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor

# Input data set name
datasetFileName = os.path.join("..", "data", "batch", "layer.csv")

if __name__ == "__main__":

    # Create collection of dimension sizes of the input data tensor
    inDims = [2, 2, 6, 8]
    tensorData = HomogenTensor(inDims, TensorIface.doAllocate, 1.0)

    # Create an algorithm to compute forward 2D locally connected layer results using default method
    locallyconnected2dLayerForward = layers.locallyconnected2d.forward.Batch()
    locallyconnected2dLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward 2D locally connected layer results
    forwardResult = locallyconnected2dLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Forward 2D locally connected layer result (first 5 rows):", 5, 15)
    printTensor(forwardResult.getLayerData(layers.locallyconnected2d.auxWeights), "2D locally connected layer weights (first 5 rows):", 5, 15)

    gDims = forwardResult.getResult(layers.forward.value).getDimensions()

    # Create input gradient tensor for backward 2D locally connected layer
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward 2D locally connected layer results using default method
    locallyconnected2dLayerBackward = layers.locallyconnected2d.backward.Batch()
    locallyconnected2dLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    locallyconnected2dLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward 2D locally connected layer results
    backwardResult = locallyconnected2dLayerBackward.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "2D locally connected layer backpropagation gradient result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives),
                "2D locally connected layer backpropagation weightDerivative result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.biasDerivatives),
                "2D locally connected layer backpropagation biasDerivative result (first 5 rows):", 5, 15)
