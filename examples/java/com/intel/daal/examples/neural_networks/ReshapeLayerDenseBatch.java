/* file: ReshapeLayerDenseBatch.java */
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
 //     Java example of Reshape layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.reshape.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-RESHAPELAYERBATCH">
 * @example ReshapeLayerDenseBatch.java
 */
class ReshapeLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor tensorData = Service.readTensorFromCSV(context, datasetFileName);

        /* Create an algorithm to compute forward reshape layer results using default method */
        ReshapeForwardBatch reshapeLayerForward = new ReshapeForwardBatch(context, Float.class, ReshapeMethod.defaultDense);

        /* Print the input data of the forward reshape layer */
        Service.printTensor("Forward reshape layer input (first 5 rows):", tensorData, 5, 0);

        /* Set input objects for the forward reshape layer */
        reshapeLayerForward.input.set(ForwardInputId.data, tensorData);

        long[] pdims = {-1, 5};
        reshapeLayerForward.parameter.setReshapeDimensions( pdims );

        /* Compute forward reshape layer results */
        ReshapeForwardResult forwardResult = reshapeLayerForward.compute();

        /* Print the results of the forward reshape layer */
        Service.printTensor("Forward reshape layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);

        /* Get the size of forward reshape layer output */
        int nSize = (int)forwardResult.get(ForwardResultId.value).getSize();
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();

        /* Create a tensor with backward input data */
        double[] data = new double[nSize];
        Tensor tensorDataBack = new HomogenTensor(context, dims, data, 0.01);

        /* Create an algorithm to compute backward reshape layer results using default method */
        ReshapeBackwardBatch reshapeLayerBackward = new ReshapeBackwardBatch(context, Float.class, ReshapeMethod.defaultDense);

        /* Set input objects for the backward reshape layer */
        reshapeLayerBackward.input.set(BackwardInputId.inputGradient, tensorDataBack);
        reshapeLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward, forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward reshape layer results */
        ReshapeBackwardResult backwardResult = reshapeLayerBackward.compute();

        /* Print the results of the backward reshape layer */
        Service.printTensor("Backward reshape layer result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 0);

        context.dispose();
    }
}
