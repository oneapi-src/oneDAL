# file: batch_norm_layer_dense_batch.py
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
# !    Python example of forward and backward batch normalization layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-BATCH_NORMALIZATION_LAYER_BATCH"></a>
## \example batch_norm_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor, TensorIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set name
datasetFileName = os.path.join("..", "data", "batch", "layer.csv")
dimension = 1

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    data = readTensorFromCSV(datasetFileName)

    printTensor(data, "Forward batch normalization layer input (first 5 rows):", 5)

    # Get collection of dimension sizes of the input data tensor
    dataDims = data.getDimensions()
    dimensionSize = dataDims[dimension]

    # Create a collection of dimension sizes of input weights, biases, population mean and variance tensors
    dimensionSizes = [dimensionSize]

    # Create input weights, biases, population mean and population variance tensors
    weights = HomogenTensor(dimensionSizes, TensorIface.doAllocate, 1.0)
    biases = HomogenTensor(dimensionSizes, TensorIface.doAllocate, 2.0)
    populationMean = HomogenTensor(dimensionSizes, TensorIface.doAllocate, 0.0)
    populationVariance = HomogenTensor(dimensionSizes, TensorIface.doAllocate, 0.0)

    # Create an algorithm to compute forward batch normalization layer results using default method
    forwardLayer = layers.batch_normalization.forward.Batch()
    forwardLayer.parameter.dimension = dimension
    forwardLayer.input.setInput(layers.forward.data, data)
    forwardLayer.input.setInput(layers.forward.weights, weights)
    forwardLayer.input.setInput(layers.forward.biases, biases)
    forwardLayer.input.setInputLayerData(layers.batch_normalization.forward.populationMean, populationMean)
    forwardLayer.input.setInputLayerData(layers.batch_normalization.forward.populationVariance, populationVariance)

    # Compute forward batch normalization layer results
    forwardResult = forwardLayer.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Forward batch normalization layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.batch_normalization.auxMean), "Mini-batch mean (first 5 values):", 5)
    printTensor(forwardResult.getLayerData(layers.batch_normalization.auxStandardDeviation), "Mini-batch standard deviation (first 5 values):", 5)
    printTensor(forwardResult.getLayerData(layers.batch_normalization.auxPopulationMean), "Population mean (first 5 values):", 5)
    printTensor(forwardResult.getLayerData(layers.batch_normalization.auxPopulationVariance), "Population variance (first 5 values):", 5)

    # Create input gradient tensor for backward batch normalization layer
    inputGradientTensor = HomogenTensor(dataDims, TensorIface.doAllocate, 10.0)

    # Create an algorithm to compute backward batch normalization layer results using default method
    backwardLayer = layers.batch_normalization.backward.Batch()
    backwardLayer.parameter.dimension = dimension
    backwardLayer.input.setInput(layers.backward.inputGradient, inputGradientTensor)
    backwardLayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward batch normalization layer results
    backwardResult = backwardLayer.compute()

    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward batch normalization layer result (first 5 rows):", 5)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives), "Weight derivatives (first 5 values):", 5)
    printTensor(backwardResult.getResult(layers.backward.biasDerivatives), "Bias derivatives (first 5 values):", 5)
