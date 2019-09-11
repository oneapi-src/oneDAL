# file: prelu_layer_dense_batch.py
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
# !    Python example of forward and backward parametric rectified linear unit (prelu) layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-PRELU_LAYER_BATCH"></a>
## \example prelu_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor, Tensor

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set parameters
datasetName = os.path.join("..", "data", "batch", "layer.csv")
weightsName = os.path.join("..", "data", "batch", "layer.csv")

dataDimension = 0
weightsDimension = 2

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    tensorData = readTensorFromCSV(datasetName)
    tensorWeights = readTensorFromCSV(weightsName)

    # Create an algorithm to compute forward prelu layer results using default method
    forwardPreluLayer = layers.prelu.forward.Batch()
    forwardPreluLayer.parameter.dataDimension = dataDimension
    forwardPreluLayer.parameter.weightsDimension = weightsDimension
    forwardPreluLayer.parameter.weightsAndBiasesInitialized = True

    # Set input objects for the forward prelu layer
    forwardPreluLayer.input.setInput(layers.forward.data, tensorData)
    forwardPreluLayer.input.setInput(layers.forward.weights, tensorWeights)

    # Compute forward prelu layer results
    forwardResult = forwardPreluLayer.compute()

    # Print the results of the forward prelu layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward prelu layer result (first 5 rows):", 5)

    # Get the size of forward prelu layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, Tensor.doAllocate, 0.01)

    # Create an algorithm to compute backward prelu layer results using default method
    backwardPreluLayer = layers.prelu.backward.Batch()
    backwardPreluLayer.parameter.dataDimension = dataDimension
    backwardPreluLayer.parameter.weightsDimension = weightsDimension

    # Set input objects for the backward prelu layer
    backwardPreluLayer.input.setInput(layers.backward.inputGradient, tensorDataBack)
    backwardPreluLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward prelu layer results
    backwardResult = backwardPreluLayer.compute()

    # Print the results of the backward prelu layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward prelu layer result (first 5 rows):", 5)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives), "Weights derivative (first 5 rows):", 5)
