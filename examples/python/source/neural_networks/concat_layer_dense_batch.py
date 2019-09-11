# file: concat_layer_dense_batch.py
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
