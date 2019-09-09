/* file: DropoutLayerDenseBatch.java */
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
 //     Java example of dropout layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.dropout.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-DROPOUTLAYERBATCH">
 * @example DropoutLayerDenseBatch.java
 */
class DropoutLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward dropout layer results using default method */
        DropoutForwardBatch forwardLayer = new DropoutForwardBatch(context, Float.class, DropoutMethod.defaultDense);

        /* Set input objects for the forward dropout layer */
        forwardLayer.input.set(ForwardInputId.data, data);

        /* Compute forward dropout layer results */
        DropoutForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward dropout layer */
        Service.printTensor("Forward dropout layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Dropout layer retain mask (first 5 rows):", forwardResult.get(DropoutLayerDataId.auxRetainMask), 5, 0);

        /* Create input gradient tensor for backward dropout layer */
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();
        double[] inputGradientData = new double[(int)forwardResult.get(ForwardResultId.value).getSize()];
        Tensor inputGradient = new HomogenTensor(context, dims, inputGradientData, 0.01);

        /* Create an algorithm to compute backward dropout layer results using default method */
        DropoutBackwardBatch backwardLayer = new DropoutBackwardBatch(context, Float.class, DropoutMethod.defaultDense);

        /* Set input objects for the backward dropout layer */
        backwardLayer.input.set(BackwardInputId.inputGradient, inputGradient);
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward dropout layer results */
        DropoutBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward dropout layer */
        Service.printTensor("Backward dropout layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
