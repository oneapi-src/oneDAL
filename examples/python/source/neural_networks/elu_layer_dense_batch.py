# file: elu_layer_dense_batch.py
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
# !    Python example of forward and backward Exponential Linear Unit (ELU) layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-ELU_LAYER_BATCH"></a>
## \example elu_layer_dense_batch.py
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

    # Create an algorithm to compute forward ELU layer results using default method
    eluLayerForward = layers.elu.forward.Batch()

    # Set input objects for the forward ELU layer
    eluLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward ELU layer results
    forwardResult = eluLayerForward.compute()

    # Print the results of the forward ELU layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward ELU layer result (first 5 rows):", 5)

    # Get the size of forward ELU layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, Tensor.doAllocate, 1.0)

    # Create an algorithm to compute backward ELU layer results using default method
    eluLayerBackward = layers.elu.backward.Batch()

    # Set input objects for the backward ELU layer
    eluLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    eluLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward ELU layer results
    backwardResult = eluLayerBackward.compute()

    # Print the results of the backward ELU layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward ELU layer result (first 5 rows):", 5)
