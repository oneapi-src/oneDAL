/* file: FullyconLayerDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
 //  Content:
 //     Java example of fully-connected layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.fullyconnected.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-FULLYCONNECTEDLAYERBATCH">
 * @example FullyconLayerDenseBatch.java
 */
class FullyconLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();
    private static final long m = 5;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward fully-connected layer results using default method */
        FullyConnectedForwardBatch forwardLayer = new FullyConnectedForwardBatch(context, Float.class, FullyConnectedMethod.defaultDense, m);

        /* Set input objects for the forward fully-connected layer */
        forwardLayer.input.set(ForwardInputId.data, data);

        /* Compute forward fully-connected layer results */
        FullyConnectedForwardResult forwardResult = forwardLayer.compute();

        /* Print the results of the forward fully-connected layer */
        Service.printTensor("Forward fully-connected layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Forward fully-connected layer weights (first 5 rows):", forwardResult.get(FullyConnectedLayerDataId.auxWeights), 5, 0);

        /* Create input gradient tensor for backward fully-connected layer */
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();
        double[] inputGradientData = new double[(int)forwardResult.get(ForwardResultId.value).getSize()];
        Tensor inputGradient = new HomogenTensor(context, dims, inputGradientData, 0.01);

        /* Create an algorithm to compute backward fully-connected layer results using default method */
        FullyConnectedBackwardBatch backwardLayer = new FullyConnectedBackwardBatch(context, Float.class, FullyConnectedMethod.defaultDense, m);

        /* Set input objects for the backward fully-connected layer */
        backwardLayer.input.set(BackwardInputId.inputGradient, inputGradient);
        backwardLayer.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward fully-connected layer results */
        FullyConnectedBackwardResult backwardResult = backwardLayer.compute();

        /* Print the results of the backward fully-connected layer */
        Service.printTensor("Backward fully-connected layer gradient result (first 5 rows):",
                            backwardResult.get(BackwardResultId.gradient), 5, 0);
        Service.printTensor("Backward fully-connected layer weightDerivative result (first 5 rows):",
                            backwardResult.get(BackwardResultId.weightDerivatives), 5, 0);
        Service.printTensor("Backward fully-connected layer biasDerivative result (first 5 rows):",
                            backwardResult.get(BackwardResultId.biasDerivatives), 5, 0);

        context.dispose();
    }
}
