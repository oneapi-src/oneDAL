/* file: InitializersDenseBatch.java */
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
 //     Java example of initializers
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.initializers.*;
import com.intel.daal.algorithms.neural_networks.initializers.gaussian.*;
import com.intel.daal.algorithms.neural_networks.initializers.uniform.*;
import com.intel.daal.algorithms.neural_networks.initializers.truncated_gaussian.*;
import com.intel.daal.algorithms.neural_networks.initializers.xavier.*;
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
 * <a name="DAAL-EXAMPLE-JAVA-INITIALIZERSDENSEBATCH">
 * @example InitializersDenseBatch.java
 */
class InitializersDenseBatch {
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create a collection of dimension sizes of input data */
        long[] dimensionSizes = new long[4];
        dimensionSizes[0] = 2;
        dimensionSizes[1] = 1;
        dimensionSizes[2] = 3;
        dimensionSizes[3] = 4;

        /* Create input daat tensor */
        float[] data = new float[24];
        Tensor dataTensor = new HomogenTensor(context, dimensionSizes, data);

        /* Fill tensor data using truncated gaussian initializer */
        /* Create an algorithm to initialize data using default method */
        TruncatedGaussianBatch truncatedGaussInitializer = new TruncatedGaussianBatch(context, Float.class, TruncatedGaussianMethod.defaultDense, 0.0, 1.0);

        /* Set input object and parameters for the truncated gaussian initializer */
        truncatedGaussInitializer.input.set(InputId.data, dataTensor);

        /* Compute truncated gaussian initializer */
        truncatedGaussInitializer.compute();

        /* Print the results of the truncated gaussian initializer */
        Service.printTensor("Data with truncated gaussian distribution:", dataTensor, 5, 0);


        /* Fill tensor data using gaussian initializer */
        /* Create an algorithm to initialize data using default method */
        GaussianBatch gaussInitializer = new GaussianBatch(context, Float.class, GaussianMethod.defaultDense, 1.0, 0.5);

        /* Set input object for the gaussian initializer */
        gaussInitializer.input.set(InputId.data, dataTensor);

        /* Compute gaussian initializer */
        gaussInitializer.compute();

        /* Print the results of the gaussian initializer */
        Service.printTensor("Data with gaussian distribution:", dataTensor, 5, 0);


        /* Fill tensor data using uniform initializer */
        /* Create an algorithm to initialize data using default method */
        UniformBatch uniformInitializer = new UniformBatch(context, Float.class, UniformMethod.defaultDense, -5.0, 5.0);

        /* Set input object and parameters for the uniform initializer */
        uniformInitializer.input.set(InputId.data, dataTensor);

        /* Compute uniform initializer */
        uniformInitializer.compute();

        /* Print the results of the uniform initializer */
        Service.printTensor("Data with uniform distribution:", dataTensor, 5, 0);


        /* Fill layer weights using xavier initializer */
        /* Create an algorithm to compute forward fully-connected layer results using default method */
        FullyConnectedForwardBatch fullyconnectedLayerForward = new FullyConnectedForwardBatch(context, Float.class, FullyConnectedMethod.defaultDense, 5);

        /* Set input objects and parameter for the forward fully-connected layer */
        fullyconnectedLayerForward.input.set(ForwardInputId.data, dataTensor);
        fullyconnectedLayerForward.parameter.setWeightsInitializer(new XavierBatch(context, Float.class, XavierMethod.defaultDense));

        /* Compute forward fully-connected layer results */
        FullyConnectedForwardResult forwardResult = fullyconnectedLayerForward.compute();

        /* Print the results of the xavier initializer */
        Service.printTensor("Weights filled by xavier initializer:", forwardResult.get(FullyConnectedLayerDataId.auxWeights), 5, 0);

        context.dispose();
    }
}
