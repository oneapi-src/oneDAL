# file: conv2d_layer_dense_batch.py
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
# !    Python example of forward and backward two-dimensional convolution layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-CONVOLUTION2D_LAYER_BATCH"></a>
## \example conv2d_layer_dense_batch.py
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
    inDims = [2, 1, 16, 16]
    tensorData = HomogenTensor(inDims, TensorIface.doAllocate, 1.0)

    # Create an algorithm to compute forward two-dimensional convolution layer results using default method
    convolution2dLayerForward = layers.convolution2d.forward.Batch()
    convolution2dLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward two-dimensional convolution layer results
    forwardResult = convolution2dLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Two-dimensional convolution layer result (first 5 rows):", 5, 15)
    printTensor(forwardResult.getLayerData(layers.convolution2d.auxWeights),
                "Two-dimensional convolution layer weights (first 5 rows):", 5, 15)

    gDims = forwardResult.getResult(layers.forward.value).getDimensions()

    # Create input gradient tensor for backward two-dimensional convolution layer
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward two-dimensional convolution layer results using default method
    convolution2dLayerBackward = layers.convolution2d.backward.Batch()
    convolution2dLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    convolution2dLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward two-dimensional convolution layer results
    backwardResult = convolution2dLayerBackward.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Two-dimensional convolution layer backpropagation gradient result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives),
                "Two-dimensional convolution layer backpropagation weightDerivative result (first 5 rows):", 5, 15)
    printTensor(backwardResult.getResult(layers.backward.biasDerivatives),
                "Two-dimensional convolution layer backpropagation biasDerivative result (first 5 rows):", 5, 15)
