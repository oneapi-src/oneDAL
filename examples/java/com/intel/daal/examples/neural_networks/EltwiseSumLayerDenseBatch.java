/* file: EltwiseSumLayerDenseBatch.java */
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
 //     Java example of element-wise sum layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.eltwise_sum.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-CONCATLAYERBATCH">
 * @example EltwiseSumLayerDenseBatch.java
 */
class EltwiseSumLayerDenseBatch {
    private static DaalContext context = new DaalContext();

    private static final String datasetFileName = "../data/batch/layer.csv";
    private static final long   nInputs = 3;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create an algorithm to compute forward concat layer results using default method */
        EltwiseSumForwardBatch forwardLayer = new EltwiseSumForwardBatch(context, Float.class, EltwiseSumMethod.defaultDense);

        /* Read datasetFileName from a file and create a tensor to store forward input data */
        for (int i = 0; i < nInputs; i++) {
            Tensor data = Service.readTensorFromCSV(context, datasetFileName);

            /* Set input objects for the forward concat layer */
            forwardLayer.input.set(ForwardInputLayerDataId.inputLayerData, data, i);
        }

        /* Compute forward concat layer results */
        EltwiseSumForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward concat layer */
        Service.printTensor("Forward element-wise sum layer result (first 5 rows):",
                            forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printNumericTable("Forward element-wise sum layer number of inputs (number of coefficients)",
                                  forwardResult.get(EltwiseSumLayerDataNumericTableId.auxNumberOfCoefficients), 1, 0);

        /* Create an algorithm to compute backward concat layer results using default method */
        EltwiseSumBackwardBatch backwardLayer = new EltwiseSumBackwardBatch(context, Float.class, EltwiseSumMethod.defaultDense);

        /* Read input gradient from CSV, for brevity use the same file as for an input data */
        Tensor inputGradient = Service.readTensorFromCSV(context, datasetFileName);

        /* Set input objects for the backward concat layer */
        backwardLayer.input.set(BackwardInputId.inputGradient, inputGradient);
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward concat layer results */
        EltwiseSumBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward concat layer */
        for (int i = 0; i < nInputs; i++) {
            Service.printTensor("Backward element-wise sum layer backward result (first 5 rows):",
                                backwardResult.get(BackwardResultLayerDataId.resultLayerData, i), 5, 0);
        }

        context.dispose();
    }
}
