# file: loss_logistic_entr_layer_dense_batch.py
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
# !    Python example of forward and backward logistic cross-entropy layer usage
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-LOSS_LOGISTIC_ENTR_LAYER_DENSE_BATCH"></a>
## \example loss_logistic_entr_layer_dense_batch.py
#

import os
import sys

from daal.algorithms.neural_networks import layers
from daal.algorithms.neural_networks.layers import loss
from daal.algorithms.neural_networks.layers.loss import logistic_cross

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printTensor, readTensorFromCSV

# Input data set parameters
datasetName = os.path.join("..", "data", "batch", "logistic_cross_entropy_layer.csv")
datasetGroundTruthName = os.path.join("..", "data", "batch", "logistic_cross_entropy_layer_ground_truth.csv")

if __name__ == "__main__":

    # Retrieve the input data
    tensorData = readTensorFromCSV(datasetName)
    groundTruth = readTensorFromCSV(datasetGroundTruthName)

    # Create an algorithm to compute forward logistic cross-entropy layer results using default method
    logisticCrossLayerForward = loss.logistic_cross.forward.Batch(method=loss.logistic_cross.defaultDense)

    # Set input objects for the forward logistic_cross layer
    logisticCrossLayerForward.input.setInput(layers.forward.data, tensorData)
    logisticCrossLayerForward.input.setInput(loss.forward.groundTruth, groundTruth)

    # Compute forward logistic_cross layer results
    forwardResult = logisticCrossLayerForward.compute()

    # Print the results of the forward logistic_cross layer
    printTensor(forwardResult.getResult(layers.forward.value), "Forward logistic cross-entropy layer result (first 5 rows):", 5)
    printTensor(forwardResult.getLayerData(loss.logistic_cross.auxGroundTruth), "Logistic Cross-Entropy layer ground truth (first 5 rows):", 5)

    # Create an algorithm to compute backward logistic_cross layer results using default method
    logisticCrossLayerBackward = logistic_cross.backward.Batch(method=loss.logistic_cross.defaultDense)

    # Set input objects for the backward logistic_cross layer
    logisticCrossLayerBackward.input.setInputLayerData(layers.backward.inputFromForward, forwardResult.getResultLayerData(layers.forward.resultForBackward))

    # Compute backward logistic_cross layer results
    backwardResult = logisticCrossLayerBackward.compute()

    # Print the results of the backward logistic_cross layer
    printTensor(backwardResult.getResult(layers.backward.gradient), "Backward logistic cross-entropy layer result (first 5 rows):", 5)
