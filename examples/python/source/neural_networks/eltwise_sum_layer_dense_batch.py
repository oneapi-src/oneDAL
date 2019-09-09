# file: eltwise_sum_layer_dense_batch.py
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
# !    Python example of forward and backward element-wise sum layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-ELTWISE_SUM_LAYER_BATCH"></a>
## \example eltwise_sum_layer_dense_batch.py
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
nInputs = 3

if __name__ == "__main__":

    # Retrieve the input data
    tensorDataCollection = layers.LayerData()
    for i in range(nInputs):
        tensorDataCollection[i] = readTensorFromCSV(datasetName)

    # Create an algorithm to compute forward element-wise sum layer results using default method
    eltwiseSumLayerForward = layers.eltwise_sum.forward.Batch()

    # Set input objects for the forward element-wise sum layer
    eltwiseSumLayerForward.input.setInputLayerData(layers.forward.inputLayerData, tensorDataCollection)

    # Compute forward element-wise sum layer results
    forwardResult = eltwiseSumLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value),
                "Forward element-wise sum layer result (first 5 rows):", 5)
    printNumericTable(forwardResult.getLayerDataNumericTable(layers.eltwise_sum.auxNumberOfCoefficients),
                      "Forward element-wise sum layer number of inputs (number of coefficients)", 1)

    # Create an algorithm to compute backward element-wise sum layer results using default method
    eltwiseSumLayerBackward = layers.eltwise_sum.backward.Batch()

    # Set inputs for the backward element-wise sum layer
    eltwiseSumLayerBackward.input.setInput(layers.backward.inputGradient, readTensorFromCSV(datasetName))
    eltwiseSumLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward element-wise sum layer results
    backwardResult = eltwiseSumLayerBackward.compute()

    for i in range(tensorDataCollection.size()):
        printTensor(backwardResult.getResultLayerData(layers.backward.resultLayerData, i),
                    "Backward element-wise sum layer backward result (first 5 rows):", 5)
