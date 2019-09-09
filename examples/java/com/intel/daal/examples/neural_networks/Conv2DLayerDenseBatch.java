/* file: Conv2DLayerDenseBatch.java */
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
 //     Java example of 2D convolution layer in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.layers.convolution2d.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-CONVOLUTION2DLAYERBATCH">
 * @example Conv2DLayerDenseBatch.java
 */
class Conv2DLayerDenseBatch {
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create a collection of dimension sizes of input data */
        long[] dimensionSizes = new long[4];
        dimensionSizes[0] = 2;
        dimensionSizes[1] = 1;
        dimensionSizes[2] = 16;
        dimensionSizes[3] = 16;

        /* Create input daat tensor */
        double[] data = new double[512];
        Tensor dataTensor = new HomogenTensor(context, dimensionSizes, data, 1.0f);

        /* Create an algorithm to compute forward 2D convolution layer results using default method */
        Convolution2dForwardBatch convolution2DLayerForward = new Convolution2dForwardBatch(context, Float.class, Convolution2dMethod.defaultDense);

        /* Set input objects for the forward 2D convolution layer */
        convolution2DLayerForward.input.set(ForwardInputId.data, dataTensor);

        /* Compute forward 2D convolution layer results */
        Convolution2dForwardResult forwardResult = convolution2DLayerForward.compute();

        /* Print the results of the forward 2D convolution layer */
        Service.printTensor("Two-dimensional convolution layer result (first 5 rows):", forwardResult.get(ForwardResultId.value), 5, 15);
        Service.printTensor("Two-dimensional convolution layer weights (first 5 rows):", forwardResult.get(Convolution2dLayerDataId.auxWeights), 5, 15);

        /* Get the size of forward 2D convolution layer output */
        int nSize = (int)forwardResult.get(ForwardResultId.value).getSize();
        long[] dims = forwardResult.get(ForwardResultId.value).getDimensions();

        /* Create a tensor with backward input data */
        double[] backData = new double[nSize];
        Tensor tensorDataBack = new HomogenTensor(context, dims, backData, 0.01f);

        /* Create an algorithm to compute backward 2D convolution layer results using default method */
        Convolution2dBackwardBatch convolution2DLayerBackward = new Convolution2dBackwardBatch(context, Float.class, Convolution2dMethod.defaultDense);

        /* Set input objects for the backward 2D convolution layer */
        convolution2DLayerBackward.input.set(BackwardInputId.inputGradient, tensorDataBack);
        convolution2DLayerBackward.input.set(BackwardInputLayerDataId.inputFromForward,
                                             forwardResult.get(ForwardResultLayerDataId.resultForBackward));

        /* Compute backward 2D convolution layer results */
        Convolution2dBackwardResult backwardResult = convolution2DLayerBackward.compute();

        /* Print the results of the backward 2D convolution layer */
        Service.printTensor("Two-dimensional convolution layer backpropagation gradient result (first 5 rows):", backwardResult.get(BackwardResultId.gradient), 5, 15);
        Service.printTensor("Two-dimensional convolution layer backpropagation weightDerivative result (first 5 rows):", backwardResult.get(BackwardResultId.weightDerivatives), 5, 15);
        Service.printTensor("Two-dimensional convolution layer backpropagation biasDerivative result (first 5 rows):", backwardResult.get(BackwardResultId.biasDerivatives), 5, 15);

        context.dispose();
    }
}
