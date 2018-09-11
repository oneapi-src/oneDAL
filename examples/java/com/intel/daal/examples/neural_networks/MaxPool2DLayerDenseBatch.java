/* file: MaxPool2DLayerDenseBatch.java */
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
 //  Java example of neural network forward and backward two-dimensional maximum pooling layers usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.maximum_pooling2d.*;
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

import com.intel.daal.data_management.data.NumericTable;
/**
 * <a name="DAAL-EXAMPLE-JAVA-MAXIMUMPOOLING2DLAYERBATCH">
 * @example MaxPool2DLayerDenseBatch.java
 */
class MaxPool2DLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        Tensor data = Service.readTensorFromCSV(context, datasetFileName);
        long nDim = data.getDimensions().length;

        /* Print the input of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional maximum pooling layer input (first 10 rows):", data, 10, 0);

        /* Create an algorithm to compute forward two-dimensional pooling results using default method */
        MaximumPooling2dForwardBatch maximumPooling2DLayerForward = new MaximumPooling2dForwardBatch(context, Float.class, MaximumPooling2dMethod.defaultDense, nDim);

        /* Set input objects for the forward two-dimensional pooling */
        maximumPooling2DLayerForward.input.set(ForwardInputId.data, data);

        /* Compute forward two-dimensional pooling results */
        MaximumPooling2dForwardResult forwardResult = maximumPooling2DLayerForward.compute();

        /* Print the results of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional maximum pooling layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Forward two-dimensional maximum pooling layer selected indices (first 10 rows):",
            forwardResult.get(MaximumPooling2dLayerDataId.auxSelectedIndices), 10, 0);

        /* Create an algorithm to compute backward two-dimensional pooling results using default method */
        MaximumPooling2dBackwardBatch maximumPooling2DLayerBackward = new MaximumPooling2dBackwardBatch(context, Float.class, MaximumPooling2dMethod.defaultDense, nDim);

        /* Set input objects for the backward two-dimensional pooling */
        maximumPooling2DLayerBackward.input.set(BackwardInputId.inputGradient, forwardResult.get(ForwardResultId.value));
        maximumPooling2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                                                forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward two-dimensional pooling results */
        MaximumPooling2dBackwardResult backwardResult = maximumPooling2DLayerBackward.compute();

        /* Print the results of the backward two-dimensional pooling */
        Service.printTensor("Backward two-dimensional maximum pooling layer result (first 10 rows):", backwardResult.get(BackwardResultId.gradient), 10, 0);

        context.dispose();
    }
}
