/* file: NeuralNetPredicConfigurator.java */
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
import com.intel.daal.algorithms.neural_networks.layers.fullyconnected.*;
import com.intel.daal.algorithms.neural_networks.layers.softmax.*;
import com.intel.daal.algorithms.neural_networks.prediction.PredictionTopology;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKPREDICTIONCONFIGURATOR">
 * @example NeuralNetPredicConfigurator.java
 */
class NeuralNetPredicConfigurator {
    public static PredictionTopology configureNet(DaalContext context) {
        /* Create layers of the neural network */
        /* Create fully-connected layer and initialize layer parameters */
        FullyConnectedForwardBatch fullyconnectedLayer1 = new FullyConnectedForwardBatch(context, Float.class, FullyConnectedMethod.defaultDense, 5);

        /* Create fully-connected layer and initialize layer parameters */
        FullyConnectedForwardBatch fullyconnectedLayer2 = new FullyConnectedForwardBatch(context, Float.class, FullyConnectedMethod.defaultDense, 2);

        /* Create softmax cross-entropy loss layer and initialize layer parameters */
        SoftmaxForwardBatch softmaxLayer = new SoftmaxForwardBatch(context, Float.class, SoftmaxMethod.defaultDense);

        /* Create topology of the neural network */
        PredictionTopology topology = new PredictionTopology(context);

        /* Add layers to the topology of the neural network */
        /* Add layers to the topology of the neural network */
        long fc1 = topology.add(fullyconnectedLayer1);
        long fc2 = topology.add(fullyconnectedLayer2);
        long sm = topology.add(softmaxLayer);
        topology.addNext(fc1, fc2);
        topology.addNext(fc2, sm);
        return topology;
    }
}
