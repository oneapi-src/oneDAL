# file: logistic_layer_dense_batch.py
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
# !    Python example of forward and backward logistic layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LOGISTIC_LAYER_BATCH"></a>
## \example logistic_layer_dense_batch.py
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

    # Retrieve the input data
    tensorData = readTensorFromCSV(datasetName)

    # Create an algorithm to compute forward logistic layer results using default method
    logisticLayerForward = layers.logistic.forward.Batch()

    # Set input objects for the forward logistic layer
    logisticLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward logistic layer results
    forwardResult = logisticLayerForward.compute()

    # Print the results of the forward logistic layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward logistic layer result (first 5 rows):", 5)

    # Get the size of forward logistic layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward logistic layer results using default method
    logisticLayerBackward = layers.logistic.backward.Batch()

    # Set input objects for the backward logistic layer
    logisticLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    logisticLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward logistic layer results
    backwardResult = logisticLayerBackward.compute()

    # Print the results of the backward logistic layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward logistic layer result (first 5 rows):", 5)
