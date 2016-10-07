/* file: SpatStochPool2DLayerDenseBatch.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
 //   Java example of neural network forward and backward two-dimensional stochastic pooling layers usage
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.spatial_stochastic_pooling2d.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-SPATSTOCHPOOL2DLAYERDENSEBATCH">
 * @example SpatStochPool2DLayerDenseBatch.java
 */
class SpatStochPool2DLayerDenseBatch {
    /* Input non-negative data set */
    private static final String datasetFileName = "../data/batch/layer_non_negative.csv";
    private static DaalContext context = new DaalContext();
    private static long pyramidHeight = 2;

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read datasetFileName from a file and create a tensor to store forward input data */
        /* Create a collection of dimension sizes of input data */
        long[] dimensionSizes = new long[4];
        dimensionSizes[0] = 2;
        dimensionSizes[1] = 1;
        dimensionSizes[2] = 16;
        dimensionSizes[3] = 16;

        /* Create input data tensor */
        double[] data = new double[512];
        Tensor dataTensor = new HomogenTensor(context, dimensionSizes, data, 1.0);

        /* Get number of dimensions in input tensor */
        long nDim = dataTensor.getDimensions().length;

        /* Print the input of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional stochastic pooling input (first 10 rows):", dataTensor, 10, 0);

        /* Create an algorithm to compute forward two-dimensional pooling results using default method */
        SpatialStochasticPooling2dForwardBatch spatialStochPooling2DLayerForward = new SpatialStochasticPooling2dForwardBatch(context, Double.class,
                                                                                                                              SpatialStochasticPooling2dMethod.defaultDense,
                                                                                                                              pyramidHeight, nDim);

        /* Set input objects for the forward two-dimensional pooling */
        spatialStochPooling2DLayerForward.input.set(ForwardInputId.data, dataTensor);

        /* Compute forward two-dimensional pooling results */
        SpatialStochasticPooling2dForwardResult forwardResult = spatialStochPooling2DLayerForward.compute();

        /* Print the results of the forward two-dimensional pooling */
        Service.printTensor("Forward two-dimensional stochastic pooling result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 0);
        Service.printTensor("Forward two-dimensional stochastic pooling layer selected indices (first 5 rows):",
                            forwardResult.get(SpatialStochasticPooling2dLayerDataId.auxSelectedIndices), 5, 0);

        /* Create an algorithm to compute backward two-dimensional pooling results using default method */
        SpatialStochasticPooling2dBackwardBatch spatialStochPooling2DLayerBackward = new SpatialStochasticPooling2dBackwardBatch(context, Double.class,
                                                                                                                                 SpatialStochasticPooling2dMethod.defaultDense,
                                                                                                                                 pyramidHeight, nDim);

        /* Set input objects for the backward two-dimensional pooling */
        spatialStochPooling2DLayerBackward.input.set(BackwardInputId.inputGradient, forwardResult.get(ForwardResultId.value));
        spatialStochPooling2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward two-dimensional pooling results */
        SpatialStochasticPooling2dBackwardResult backwardResult = spatialStochPooling2DLayerBackward.compute();

        /* Print the results of the backward two-dimensional pooling */
        Service.printTensor("Backward two-dimensional stochastic pooling result (first 10 rows):", backwardResult.get(BackwardResultId.gradient), 10, 0);

        context.dispose();
    }
}
