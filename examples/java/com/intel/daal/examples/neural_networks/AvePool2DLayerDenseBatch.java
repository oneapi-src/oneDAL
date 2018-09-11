/* file: AvePool2DLayerDenseBatch.java */
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
 //  Java example of neural network forward and backward two-dimensional average pooling layers usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.average_pooling2d.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-AVERAGEPOOLING2DLAYERBATCH">
 * @example AvePool2DLayerDenseBatch.java
 */
class AvePool2DLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);
        long nDim = data.getDimensions().length;

        /* Print the input of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional average pooling layer input (first 10 rows):", data, 10, 0);

        /* Create an algorithm to compute forward two-dimensional pooling layer results using average method */
        AveragePooling2dForwardBatch averagePooling2DLayerForward = new AveragePooling2dForwardBatch(context, Float.class, AveragePooling2dMethod.defaultDense, nDim);

        /* Set input objects for the forward two-dimensional pooling */
        averagePooling2DLayerForward.input.set(ForwardInputId.data, data);

        /* Compute forward two-dimensional pooling results */
        AveragePooling2dForwardResult forwardResult = averagePooling2DLayerForward.compute();

        /* Print the results of the forward two-dimensional average pooling layer */
        Service.printTensor("Forward two-dimensional average pooling layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printNumericTable("Forward two-dimensional average pooling layer input dimensions:",
                                  forwardResult.get(AveragePooling2dLayerDataId.auxInputDimensions));

        /* Create an algorithm to compute backward two-dimensional pooling layer results using average method */
        AveragePooling2dBackwardBatch averagePooling2DLayerBackward = new AveragePooling2dBackwardBatch(context, Float.class, AveragePooling2dMethod.defaultDense, nDim);

        /* Set input objects for the backward two-dimensional average pooling layer */
        averagePooling2DLayerBackward.input.set(BackwardInputId.inputGradient, forwardResult.get(ForwardResultId.value));
        averagePooling2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                                                forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward two-dimensional pooling results */
        AveragePooling2dBackwardResult backwardResult = averagePooling2DLayerBackward.compute();

        /* Print the results of the backward two-dimensional average pooling layer */
        Service.printTensor("Backward two-dimensional average pooling layer result (first 10 rows):", backwardResult.get(BackwardResultId.gradient), 10, 0);

        context.dispose();
    }
}
