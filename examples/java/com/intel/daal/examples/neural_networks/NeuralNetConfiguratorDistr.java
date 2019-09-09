/* file: NeuralNetConfiguratorDistr.java */
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
 //     Java example of neural network configurator
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.*;
import com.intel.daal.algorithms.neural_networks.initializers.uniform.*;
import com.intel.daal.algorithms.neural_networks.training.TrainingTopology;
import com.intel.daal.algorithms.neural_networks.layers.fullyconnected.*;
import com.intel.daal.algorithms.neural_networks.layers.softmax_cross.*;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKCONFIGURATORDISTR">
 * @example NeuralNetConfiguratorDistr.java
 */
class NeuralNetConfiguratorDistr {
    public static TrainingTopology configureNet(DaalContext context) {
        /* Create layers of the neural network */
        /* Create fully-connected layer and initialize layer parameters */
        FullyConnectedBatch fullyconnectedLayer1 = new FullyConnectedBatch(context, Float.class, FullyConnectedMethod.defaultDense, 20);

        fullyconnectedLayer1.parameter.setWeightsInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, -0.001, 0.001));

        fullyconnectedLayer1.parameter.setBiasesInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, 0, 0.5));

        /* Create fully-connected layer and initialize layer parameters */
        FullyConnectedBatch fullyconnectedLayer2 = new FullyConnectedBatch(context, Float.class, FullyConnectedMethod.defaultDense, 40);

        fullyconnectedLayer2.parameter.setWeightsInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, 0.5, 1));

        fullyconnectedLayer2.parameter.setBiasesInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, 0.5, 1));

        /* Create fully-connected layer and initialize layer parameters */
        FullyConnectedBatch fullyconnectedLayer3 = new FullyConnectedBatch(context, Float.class, FullyConnectedMethod.defaultDense, 2);

        fullyconnectedLayer3.parameter.setWeightsInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, -0.005, 0.005));

        fullyconnectedLayer3.parameter.setBiasesInitializer(new UniformBatch(context, Float.class, UniformMethod.defaultDense, 0, 1));

        /* Create softmax cross-entropy loss layer and initialize layer parameters */
        SoftmaxCrossBatch softmaxCrossEntropyLayer = new SoftmaxCrossBatch(context, Float.class, SoftmaxCrossMethod.defaultDense);

        /* Create topology of the neural network */
        TrainingTopology topology = new TrainingTopology(context);

        /* Add layers to the topology of the neural network */
        long fc1 = topology.add(fullyconnectedLayer1);
        long fc2 = topology.add(fullyconnectedLayer2);
        long fc3 = topology.add(fullyconnectedLayer3);
        long sm = topology.add(softmaxCrossEntropyLayer);
        topology.addNext(fc1, fc2);
        topology.addNext(fc2, fc3);
        topology.addNext(fc3, sm);
        return topology;
    }
}
