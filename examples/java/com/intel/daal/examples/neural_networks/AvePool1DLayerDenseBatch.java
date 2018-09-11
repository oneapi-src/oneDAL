/* file: AvePool1DLayerDenseBatch.java */
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
 //  Java example of neural network forward and backward one-dimensional average pooling layers usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.average_pooling1d.*;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultLayerDataId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputLayerDataId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-AVERAGEPOOLING1DLAYERBATCH">
 * @example AvePool1DLayerDenseBatch.java
 */
class AvePool1DLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);
        long nDim = data.getDimensions().length;

        /* Print the input of the forward one-dimensional pooling */
        Service.printTensor("Forward one-dimensional average pooling layer input (first 10 rows):", data, 10, 0);

        /* Create an algorithm to compute forward one-dimensional pooling layer results using average method */
        AveragePooling1dForwardBatch averagePooling1DLayerForward = new AveragePooling1dForwardBatch(context, Float.class, AveragePooling1dMethod.defaultDense, nDim);

        /* Set input objects for the forward one-dimensional pooling */
        averagePooling1DLayerForward.input.set(ForwardInputId.data, data);

        /* Compute forward one-dimensional pooling results */
        AveragePooling1dForwardResult forwardResult = averagePooling1DLayerForward.compute();

        /* Print the results of the forward one-dimensional average pooling layer */
        Service.printTensor("Forward one-dimensional average pooling layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printNumericTable("Forward one-dimensional average pooling layer input dimensions:",
                                  forwardResult.get(AveragePooling1dLayerDataId.auxInputDimensions));

        /* Create an algorithm to compute backward one-dimensional pooling layer results using average method */
        AveragePooling1dBackwardBatch averagePooling1DLayerBackward = new AveragePooling1dBackwardBatch(context, Float.class, AveragePooling1dMethod.defaultDense, nDim);

        /* Set input objects for the backward one-dimensional average pooling layer */
        averagePooling1DLayerBackward.input.set(BackwardInputId.inputGradient, forwardResult.get(ForwardResultId.value));
        averagePooling1DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                                                forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward one-dimensional pooling results */
        AveragePooling1dBackwardResult backwardResult = averagePooling1DLayerBackward.compute();

        /* Print the results of the backward one-dimensional average pooling layer */
        Service.printTensor("Backward one-dimensional average pooling layer result (first 10 rows):", backwardResult.get(BackwardResultId.gradient), 10, 0);

        context.dispose();
    }
}
