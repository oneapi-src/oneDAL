# file: relu_layer_dense_batch.py
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
# !    Python example of forward and backward rectified linear unit (relu) layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-RELU_LAYER_BATCH"></a>
## \example relu_layer_dense_batch.py
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

if __name__ == "__main__":

    # Read datasetFileName from a file and create a tensor to store input data
    tensorData = readTensorFromCSV(datasetName)

    # Create an algorithm to compute forward relu layer results using default method
    reluLayerForward = layers.relu.forward.Batch()

    # Set input objects for the forward relu layer
    reluLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward relu layer results
    forwardResult = reluLayerForward.compute()

    # Print the results of the forward relu layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward relu layer result (first 5 rows):", 5)

    # Get the size of forward relu layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, Tensor.doAllocate, 0.01)

    # Create an algorithm to compute backward relu layer results using default method
    reluLayerBackward = layers.relu.backward.Batch()

    # Set input objects for the backward relu layer
    reluLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    reluLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward relu layer results
    backwardResult = reluLayerBackward.compute()

    # Print the results of the backward relu layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward relu layer result (first 5 rows):", 5)
