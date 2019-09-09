# file: fullycon_layer_dense_batch.py
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
# !    Python example of forward and backward fully-connected layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-FULLYCONNECTED_LAYER_BATCH"></a>
## \example fullycon_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.data_management import HomogenTensor, TensorIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set parameters
datasetName = os.path.join("..", "data", "batch", "layer.csv")

if __name__ == "__main__":

    k = 0
    m = 5
    # Read datasetFileName from a file and create a tensor to store input data
    tensorData = readTensorFromCSV(datasetName)

    # Create an algorithm to compute forward fully-connected layer results using default method
    fullyconnectedLayerForward = layers.fullyconnected.forward.Batch(m)
    fullyconnectedLayerForward.parameter.dim = k

    # Set input objects for the forward fully-connected layer
    fullyconnectedLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward fully-connected layer results
    forwardResult = fullyconnectedLayerForward.compute()

    # Print the results of the forward fully-connected layer
    printTensor(forwardResult.getResult(layers.forward.value),
                "Forward fully-connected layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.fullyconnected.auxWeights),
                "Forward fully-connected layer weights (first 5 rows):", 5)

    # Get the size of forward fully-connected layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward fully-connected layer results using default method
    fullyconnectedLayerBackward = layers.fullyconnected.backward.Batch(m)

    # Set input objects for the backward fully-connected layer
    fullyconnectedLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    fullyconnectedLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward fully-connected layer results
    backwardResult = fullyconnectedLayerBackward.compute()

    # Print the results of the backward fully-connected layer
    printTensor(backwardResult.getResult(layers.backward.gradient),
                "Backward fully-connected layer gradient result (first 5 rows):", 5)
    printTensor(backwardResult.getResult(layers.backward.weightDerivatives),
                "Backward fully-connected layer weightDerivative result (first 5 rows):", 5)
    printTensor(backwardResult.getResult(layers.backward.biasDerivatives),
                "Backward fully-connected layer biasDerivative result (first 5 rows):", 5)
