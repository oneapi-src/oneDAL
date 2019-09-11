# file: lcn_layer_dense_batch.py
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
# !    Python example of forward and backward local contrast normalization layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LCN_LAYER_BATCH"></a>
## \example lcn_layer_dense_batch.py
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
    inDims = [2, 1, 3, 4]
    tensorData = HomogenTensor(inDims, TensorIface.doAllocate, 1.0)

    # Create an algorithm to compute forward two-dimensional convolution layer results using default method
    lcnLayerForward = layers.lcn.forward.Batch()
    lcnLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward two-dimensional convolution layer results
    forwardResult = lcnLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value),          "Forward local contrast normalization layer result:")
    printTensor(forwardResult.getLayerData(layers.lcn.auxCenteredData), "Centered data tensor:")
    printTensor(forwardResult.getLayerData(layers.lcn.auxSigma),        "Sigma tensor:")
    printTensor(forwardResult.getLayerData(layers.lcn.auxC),            "C tensor:")
    printTensor(forwardResult.getLayerData(layers.lcn.auxInvMax),       "Inverted max(sigma, C):")

    # Create input gradient tensor for backward two-dimensional convolution layer
    tensorDataBack = HomogenTensor(inDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward two-dimensional convolution layer results using default method
    lcnLayerBackward = layers.lcn.backward.Batch()
    lcnLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    lcnLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward two-dimensional convolution layer results
    backwardResult = lcnLayerBackward.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Local contrast normalization layer backpropagation gradient result:")
