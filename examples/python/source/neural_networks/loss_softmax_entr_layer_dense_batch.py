# file: loss_softmax_entr_layer_dense_batch.py
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
# !    Python example of forward and backward softmax cross-entropy layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LOSS_SOFTMAX_ENTR_LAYER_DENSE_BATCH"></a>
## \example loss_softmax_entr_layer_dense_batch.py
#

import os
import sys
from daal.data_management import HomogenTensor
from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import loss
from daal.algorithms.neural_networks.layers.loss import softmax_cross

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set parameters
datasetGroundTruth = [[[1, 0, 0, 1]],[[0, 0, 1, 1]],[[1, 0, 0, 1]]];
dataset = [[[ 1,  2,  3,  4],[ 5,  6,  7,  8]],[[9, 10, 11, 12],[13, 14, 15, 16]],[[17, 18, 19, 20],[21, 22, 23, 24]]];


if __name__ == "__main__":

    # Retrieve the input data
    groundTruth = HomogenTensor(datasetGroundTruth)
    tensorData = HomogenTensor(dataset)

    printTensor(tensorData, "Forward softmax cross-entropy layer input data:");
    printTensor(groundTruth, "Forward softmax cross-entropy layer input ground truth:");

    # Create an algorithm to compute forward softmax cross-entropy layer results using default method
    softmaxCrossLayerForward = loss.softmax_cross.forward.Batch(method=loss.softmax_cross.defaultDense)

    # Set input objects for the forward softmax_cross layer
    softmaxCrossLayerForward.input.setInput(layers.forward.data, tensorData)
    softmaxCrossLayerForward.input.setInput(loss.forward.groundTruth, groundTruth)

    # Compute forward softmax_cross layer results
    forwardResult = softmaxCrossLayerForward.compute()

    # Print the results of the forward softmax_cross layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward softmax cross-entropy layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(loss.softmax_cross.auxProbabilities), "Softmax Cross-Entropy layer probabilities estimations (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(loss.softmax_cross.auxGroundTruth), "Softmax Cross-Entropy layer ground truth (first 5 rows):", 5)

    # Create an algorithm to compute backward softmax_cross layer results using default method
    softmaxCrossLayerBackward = softmax_cross.backward.Batch(method=loss.softmax_cross.defaultDense)

    # Set input objects for the backward softmax_cross layer
    softmaxCrossLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward softmax_cross layer results
    backwardResult = softmaxCrossLayerBackward.compute()

    # Print the results of the backward softmax_cross layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward softmax cross-entropy layer result (first 5 rows):", 5)
