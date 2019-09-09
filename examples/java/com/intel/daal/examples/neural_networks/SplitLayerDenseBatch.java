/* file: SplitLayerDenseBatch.java */
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
 //     Java example of split layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.split.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-SPLITLAYERBATCH">
 * @example SplitLayerDenseBatch.java
 */
class SplitLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();
    private static final long nOutputs = 3;
    private static final long nInputs  = 3;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward split layer results using default method */
        SplitForwardBatch forwardLayer = new SplitForwardBatch(context, Float.class, SplitMethod.defaultDense);

        /* Set parameters for the forward split layer */
        forwardLayer.parameter.setNOutputs(nOutputs);
        forwardLayer.parameter.setNInputs(nInputs);

        /* Set input objects for the forward split layer */
        forwardLayer.input.set(ForwardInputId.data, data);

        Service.printTensor("Split layer input (first 5 rows):", data, 5, 0);

        /* Compute forward split layer results */
        SplitForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward split layer */
        for (int i = 0; i < (int)nOutputs; i++) {
            Service.printTensor("Forward split layer result (first 5 rows):", forwardResult.get(SplitForwardResultLayerDataId.valueCollection, i), 5, 0);
        }

        /* Create an algorithm to compute backward split layer results using default method */
        SplitBackwardBatch backwardLayer = new SplitBackwardBatch(context, Float.class, SplitMethod.defaultDense);

        /* Set parameters for the backward split layer */
        backwardLayer.parameter.setNOutputs(nOutputs);
        backwardLayer.parameter.setNInputs(nInputs);

        /* Set input objects for the backward split layer */
        backwardLayer.input.set(SplitBackwardInputLayerDataId.inputGradientCollection, forwardResult.get(SplitForwardResultLayerDataId.valueCollection));

        /* Compute backward split layer results */
        SplitBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward split layer */
        Service.printTensor("Backward split layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
