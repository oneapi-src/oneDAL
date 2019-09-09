/* file: SpatAvePool2DLayerDenseBatch.java */
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
 //  Java example of neural network forward and backward two-dimensional average pooling layers usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.spatial_average_pooling2d.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-SPATAVEPOOL2DLAYERDENSEBATCH">
 * @example SpatAvePool2DLayerDenseBatch.java
 */
class SpatAvePool2DLayerDenseBatch {
    private static final String datasetFileName = "../data/batch/layer.csv";
    private static DaalContext context = new DaalContext();
    private static long pyramidHeight = 2;

    static float dataArray[] = {
        2, 4, 6, 8,
        10, 12, 14, 16,
        18, 20, 22, 24,
        26, 28, 30, 32,
        34, 36, 38, 40,
        42, 44, 46, 48,
        -2, -4, -6, -8,
        -10, -12, -14, -16,
        - 18, -20, -22, -24,
        -26, -28, -30, -32,
        - 34, -36, -38, -40,
        -42, -44, -46, -48
    };
    static long[] dims = {2, 3, 2, 4};

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store input data */

        HomogenTensor dataTensor = new HomogenTensor(context, dims, dataArray);

        long nDim = dataTensor.getDimensions().length;

        /* Print the input of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional spatial pyramid average pooling layer input (first 10 rows):", dataTensor, 10, 0);

        /* Create an algorithm to compute forward two-dimensional pooling layer results using average method */
        SpatialAveragePooling2dForwardBatch spatialAvePooling2DLayerForward = new SpatialAveragePooling2dForwardBatch(context, Float.class,
                                                                                                                      SpatialAveragePooling2dMethod.defaultDense,
                                                                                                                      pyramidHeight, nDim);

        /* Set input objects for the forward two-dimensional pooling */
        spatialAvePooling2DLayerForward.input.set(ForwardInputId.data, dataTensor);

        /* Compute forward two-dimensional pooling results */
        SpatialAveragePooling2dForwardResult forwardResult = spatialAvePooling2DLayerForward.compute();

        /* Print the results of the forward two-dimensional average pooling layer */
        Service.printTensor("Forward two-dimensional spatial pyramid average pooling layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printNumericTable("Forward two-dimensional spatial pyramid average pooling layer input dimensions:",
                                  forwardResult.get(SpatialAveragePooling2dLayerDataId.auxInputDimensions));

        /* Create an algorithm to compute backward two-dimensional pooling layer results using average method */
        SpatialAveragePooling2dBackwardBatch spatialAvePooling2DLayerBackward = new SpatialAveragePooling2dBackwardBatch(context, Float.class,
                                                                                                                         SpatialAveragePooling2dMethod.defaultDense,
                                                                                                                         pyramidHeight, nDim);

        /* Set input objects for the backward two-dimensional average pooling layer */
        spatialAvePooling2DLayerBackward.input.set(BackwardInputId.inputGradient, forwardResult.get(ForwardResultId.value));
        spatialAvePooling2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward two-dimensional pooling results */
        SpatialAveragePooling2dBackwardResult backwardResult = spatialAvePooling2DLayerBackward.compute();

        /* Print the results of the backward two-dimensional average pooling layer */
        Service.printTensor("Backward two-dimensional spatial pyramid average pooling layer result (first 10 rows):", backwardResult.get(BackwardResultId.gradient), 10, 0);

        context.dispose();
    }
}
