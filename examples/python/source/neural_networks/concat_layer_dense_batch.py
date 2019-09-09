# file: concat_layer_dense_batch.py
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
# !    Python example of forward and backward concatenation (concat) layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-CONCAT_LAYER_BATCH"></a>
## \example concat_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, printTensor, readTensorFromCSV

# Input data set parameters
datasetName = os.path.join("..", "data", "batch", "layer.csv")
concatDimension = 1
nInputs = 3

if __name__ == "__main__":

    # Retrieve the input data
    tensorData = readTensorFromCSV(datasetName)
    tensorDataCollection = layers.LayerData()

    for i in range(nInputs):
        tensorDataCollection[i] = tensorData

    # Create an algorithm to compute forward concatenation layer results using default method
    concatLayerForward = layers.concat.forward.Batch(concatDimension)

    # Set input objects for the forward concatenation layer
    concatLayerForward.input.setInputLayerData(layers.forward.inputLayerData, tensorDataCollection)

    # Compute forward concatenation layer results
    forwardResult = concatLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Forward concatenation layer result value (first 5 rows):", 5)

    # Create an algorithm to compute backward concatenation layer results using default method
    concatLayerBackward = layers.concat.backward.Batch(concatDimension)

    # Set inputs for the backward concatenation layer
    concatLayerBackward.input.setInput(layers.backward.inputGradient, forwardResult.getResult(layers.forward.value))
    concatLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    printNumericTable(forwardResult.getLayerData(layers.concat.auxInputDimensions), "auxInputDimensions ")

    # Compute backward concatenation layer results
    backwardResult = concatLayerBackward.compute()

    for i in range(tensorDataCollection.size()):
        printTensor(backwardResult.getResultLayerData(layers.backward.resultLayerData, i),
                    "Backward concatenation layer backward result (first 5 rows):", 5)
