/* file: NeuralNetConfiguratorDistr.java */
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
