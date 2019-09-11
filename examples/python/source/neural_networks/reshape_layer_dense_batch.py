# file: reshape_layer_dense_batch.py
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
# !    Python example of forward and backward reshape layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-RESHAPE_LAYER_BATCH"></a>
## \example reshape_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import reshape
from daal.data_management import HomogenTensor, Tensor

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set parameters
datasetName = os.path.join("..", "data", "batch", "layer.csv")

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    tensorData = readTensorFromCSV(datasetName)

    reshapeDimensions = [-1, 5]

    # Create an algorithm to compute forward reshape layer results using default method
    reshapeLayerForward = reshape.forward.Batch(reshapeDimensions)

    # Set input objects for the forward reshape layer
    reshapeLayerForward.input.setInput(layers.forward.data, tensorData)

    printTensor(tensorData, "Forward reshape layer input (first 5 rows):", 5)

    # Compute forward reshape layer results
    forwardResult = reshapeLayerForward.compute()

    # Print the results of the forward reshape layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward reshape layer result (first 5 rows):", 5)

    # Get the size of forward reshape layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, Tensor.doAllocate, 0.01)

    # Create an algorithm to compute backward reshape layer results using default method
    reshapeLayerBackward = reshape.backward.Batch()

    # Set input objects for the backward reshape layer
    reshapeLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    reshapeLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward reshape layer results
    backwardResult = reshapeLayerBackward.compute()

    # Print the results of the backward reshape layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward reshape layer result (first 5 rows):", 5)
