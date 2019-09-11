# file: dropout_layer_dense_batch.py
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
# !    Python example of forward and backward dropout layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-DROPOUT_LAYER_BATCH"></a>
## \example dropout_layer_dense_batch.py
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

    # Read datasetFileName from a file and create a tensor to store input data
    tensorData = readTensorFromCSV(datasetName)

    # Create an algorithm to compute forward dropout layer results using default method
    dropoutLayerForward = layers.dropout.forward.Batch()

    # Set input objects for the forward dropout layer
    dropoutLayerForward.input.setInput(layers.forward.data, tensorData)

    # Compute forward dropout layer results
    forwardResult = dropoutLayerForward.compute()

    printTensor(forwardResult.getResult(layers.forward.value), "Forward dropout layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.dropout.auxRetainMask), "Dropout layer retain mask (first 5 rows):", 5)

    # Get the size of forward dropout layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward dropout layer results using default method
    dropoutLayerBackward = layers.dropout.backward.Batch()

    # Set input objects for the backward dropout layer
    dropoutLayerBackward.input.setInput(layers.backward.inputGradient, tensorDataBack)
    dropoutLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward dropout layer results
    backwardResult = dropoutLayerBackward.compute()

    # Print the results of the backward dropout layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward dropout layer result (first 5 rows):", 5)
