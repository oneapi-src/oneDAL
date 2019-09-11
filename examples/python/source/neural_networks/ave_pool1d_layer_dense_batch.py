# file: ave_pool1d_layer_dense_batch.py
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
# !    Python example of neural network forward and backward one-dimensional average pooling layers usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-AVERAGE_POOLING1D_LAYER_BATCH"></a>
## \example ave_pool1d_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV,printNumericTable

# Input data set name
datasetFileName = os.path.join("..", "data", "batch", "layer.csv")

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    data = readTensorFromCSV(datasetFileName)
    nDim = data.getNumberOfDimensions()

    printTensor(data, "Forward one-dimensional average pooling layer input (first 10 rows):", 10)

    # Create an algorithm to compute forward one-dimensional pooling layer results using average method
    forwardLayer = layers.average_pooling1d.forward.Batch(nDim)
    forwardLayer.input.setInput(layers.forward.data, data)

    # Compute forward one-dimensional average pooling layer results
    # Result class from layers.average_pooling1d.forward
    forwardResult = forwardLayer.compute()

    # Print the results of the forward one-dimensional average pooling layer
    printTensor(forwardResult.getResult(layers.forward.value),
                "Forward one-dimensional average pooling layer result (first 5 rows):",
                5)
    printNumericTable(forwardResult.getLayerData(layers.average_pooling1d.auxInputDimensions),
                      "Forward one-dimensional average pooling layer input dimensions:")

    # Create an algorithm to compute backward one-dimensional pooling layer results using average method
    backwardLayer = layers.average_pooling1d.backward.Batch(nDim)

    # Set input objects for the backward one-dimensional average pooling layer
    backwardLayer.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward one-dimensional average pooling layer results
    # Result class from layers.average_pooling1d.backward
    backwardResult = backwardLayer.compute()

    # Print the results of the backward one-dimensional average pooling layer
    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Backward one-dimensional average pooling layer result (first 10 rows):",
                10)
