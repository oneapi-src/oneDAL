# file: lrn_layer_dense_batch.py
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
# !    Python example of forward and backward local response normalization (lrn) layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LRN_LAYER_BATCH"></a>
## \example lrn_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import lrn
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

    # Create an algorithm to compute forward local response normalization layer results using default method
    forwardLRNlayer = lrn.forward.Batch()

    # Set input objects for the forward local response normalization layer
    forwardLRNlayer.input.setInput(layers.forward.data, tensorData)

    # Compute forward local response normalization layer results
    forwardResult = forwardLRNlayer.compute()

    # Print the results of the forward local response normalization layer
    printTensor(tensorData, "LRN layer input (first 5 rows):", 5)
    printTensor(forwardResult.getResult(layers.forward.value), "LRN layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(layers.lrn.auxSmBeta), "LRN layer auxSmBeta (first 5 rows):", 5)

    # Get the size of forward local response normalization layer output
    gDims = forwardResult.getResult(layers.forward.value).getDimensions()
    tensorDataBack = HomogenTensor(gDims, TensorIface.doAllocate, 0.01)

    # Create an algorithm to compute backward local response normalization layer results using default method
    backwardLRNlayer = lrn.backward.Batch()

    # Set input objects for the backward local response normalization layer
    backwardLRNlayer.input.setInput(layers.backward.inputGradient, tensorDataBack)
    backwardLRNlayer.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward local response normalization layer results
    backwardResult = backwardLRNlayer.compute()

    # Print the results of the backward local response normalization layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "LRN layer backpropagation result (first 5 rows):", 5)
