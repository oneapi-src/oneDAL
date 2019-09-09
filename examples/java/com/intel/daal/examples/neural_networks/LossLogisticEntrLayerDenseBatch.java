/* file: LossLogisticEntrLayerDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of logistic cross-entropy layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.logistic_cross.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.loss.LossForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-LOSSLOGISTICCROSSENTROPYLAYERBATCH">
 * @example LossLogisticEntrLayerDenseBatch.java
 */
class LossLogisticEntrLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/logistic_cross_entropy_layer.csv";
    private static final String datasetGroundTruthFileName = "../data/batch/logistic_cross_entropy_layer_ground_truth.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);
        Tensor groundTruth = Service.readTensorFromCSV(context, datasetGroundTruthFileName);

        /* Create an algorithm to compute forward logistic cross-entropy layer results using default method */
        LogisticCrossForwardBatch forwardLayer = new LogisticCrossForwardBatch(context, Float.class, LogisticCrossMethod.defaultDense);

        /* Set input objects for the forward logistic cross-entropy layer */
        forwardLayer.input.set(LossForwardInputId.data, data);
        forwardLayer.input.set(LossForwardInputId.groundTruth, groundTruth);

        /* Compute forward logistic cross-entropy layer results */
        LogisticCrossForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward logistic cross-entropy layer */
        Service.printTensor("Forward logistic cross-entropy layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Logistic cross-Entropy layer probabilities estimations (first 5 rows):", forwardResult.get(LogisticCrossLayerDataId.auxGroundTruth), 5, 0);

        /* Create an algorithm to compute backward logistic cross-entropy layer results using default method */
        LogisticCrossBackwardBatch backwardLayer = new LogisticCrossBackwardBatch(context, Float.class, LogisticCrossMethod.defaultDense);

        /* Set input objects for the backward logistic cross-entropy layer */
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward logistic cross-entropy layer results */
        LogisticCrossBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward logistic cross-entropy layer */
        Service.printTensor("Backward logistic cross-entropy layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
