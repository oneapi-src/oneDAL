# file: trans_conv2d_layer_dense_batch.py
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
# !    Python example of forward and backward two-dimensional transposed convolution layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-TRANS_CONV2D_LAYER_DENSE_BATCH"></a>
## \example trans_conv2d_layer_dense_batch.py
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
    inDims = [1, 2, 4, 4]
    tensorData = HomogenTensor(inDims, TensorIface.doAllocate, 1.0)

    # Create an algorithm to compute forward two-dimensional transposed convolution layer results using default method
    transposedConv2dLayerForward = layers.transposed_conv2d.forward.Batch()
    transposedConv2dLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward two-dimensional transposed convolution layer results
    forwardResult = transposedConv2dLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Two-dimensional transposed convolution layer result (first 5 rows):", 5, 15)
    printTensor(forwardResult.getLayerData(layers.transposed_conv2d.auxWeights),
                "Two-dimensional transposed convolution layer weights (first 5 rows):", 5, 15)

    gDims = forwardResult.getResult(layers.forward.value).getDimensions()

    # Create input gradient tensor for backward two-dimensional transposed convolution layer
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward two-dimensional transposed convolution layer results using default method
    transposedConv2dLayerBackward = layers.transposed_conv2d.backward.Batch()
    transposedConv2dLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    transposedConv2dLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward two-dimensional transposed convolution layer results
    backwardResult = transposedConv2dLayerBackward.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Two-dimensional transposed convolution layer backpropagation gradient result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives),
                "Two-dimensional transposed convolution layer backpropagation weightDerivative result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.biasDerivatives),
                "Two-dimensional transposed convolution layer backpropagation biasDerivative result (first 5 rows):", 5, 15)
