/* file: LossSoftmaxEntrLayerDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 //     Java example of softmax cross-entropy layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.softmax_cross.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-LOSSSOFTMAXCROSSENTROPYLAYERBATCH">
 * @example LossSoftmaxEntrLayerDenseBatch.java
 */
class LossSoftmaxEntrLayerDenseBatch {

    private static DaalContext context = new DaalContext();
    static double dataArray[] = {
        2.74, 0.81, 3.03, 3.04, 0.27, 0.23, 3.31, 2.32,
        3.16, 3.64, 1.30, 0.65, 1.76, 2.72, 4.93, 4.91,
        1.71, 0.85, 3.47, 0.33, 0.76, 2.25, 0.34, 1.16,
        3.00, 3.81, 3.30, 3.05, 4.94, 4.32, 3.14, 3.95,
        3.52, 4.57, 4.93, 3.54, 3.68, 1.98, 2.52, 0.81,
        4.97, 1.42, 4.98, 2.24, 0.07, 4.66, 2.19, 3.06,
        3.37, 3.05, 0.64, 3.83, 1.53, 1.57, 2.16, 4.45,
        2.21, 2.87, 1.39, 0.82, 2.39, 0.30, 3.38, 3.66,
        0.67, 3.27, 2.95, 1.78, 0.62, 3.64, 1.56, 0.80,
        3.42, 2.18, 2.92, 3.77, 4.91, 2.66, 4.95, 1.13,
        1.52, 3.22, 4.18, 1.27, 2.11, 1.40, 0.72, 4.55,
        0.48, 2.75, 4.74, 0.49, 0.70, 0.53, 4.56, 3.46,
        2.69, 2.33, 3.02, 2.98
    };
    static float groundTruthArray[] = {
        3, 4, 4, 0,
        4, 3, 4, 1,
        2, 0, 1, 0,
        3, 4, 2, 3,
        2, 2, 0, 0
    };
    static long[] dims = {5, 5, 4};
    static long[] gtDims = {5, 1, 4};

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        HomogenTensor data = new HomogenTensor(context, dims, dataArray);
        HomogenTensor groundTruth = new HomogenTensor(context, gtDims, groundTruthArray);

        /* Print the input of the forward softmax cross-entropy layer */
        Service.printTensor("Forward softmax cross-entropy layer input data:", data, 0, 0);
        Service.printTensor("Forward softmax cross-entropy layer input ground truths:", groundTruth, 0, 0);

        /* Create an algorithm to compute forward softmax cross-entropy layer results using default method */
        SoftmaxCrossForwardBatch forwardLayer = new SoftmaxCrossForwardBatch(context, Float.class, SoftmaxCrossMethod.defaultDense);

        /* Set input objects for the forward softmax cross-entropy layer */
        forwardLayer.input.set(LossForwardInputId.data, data);
        forwardLayer.input.set(LossForwardInputId.groundTruth, groundTruth);

        /* Compute forward softmax cross-entropy layer results */
        SoftmaxCrossForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward softmax cross-entropy layer */
        Service.printTensor("Forward softmax cross-entropy layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Forward softmax cross-entropy layer probabilities estimations (first 5 rows):", forwardResult.get(SoftmaxCrossLayerDataId.auxProbabilities), 5, 0);

        /* Create an algorithm to compute backward softmax cross-entropy layer results using default method */
        SoftmaxCrossBackwardBatch backwardLayer = new SoftmaxCrossBackwardBatch(context, Float.class, SoftmaxCrossMethod.defaultDense);

        /* Set input objects for the backward softmax cross-entropy layer */
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward softmax cross-entropy layer results */
        SoftmaxCrossBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward softmax cross-entropy layer */
        Service.printTensor("Backward softmax cross-entropy layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
