/* file: NeuralNetDenseDistr.java */
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
 //     Java example of neural network in the distributed processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.*;
import com.intel.daal.algorithms.neural_networks.prediction.*;
import com.intel.daal.algorithms.neural_networks.training.*;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.ForwardResultId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardInputId;
import com.intel.daal.algorithms.neural_networks.layers.BackwardResultId;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.HomogenTensor;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKDISTR">
 * @example NeuralNetDenseDistr.java
 */
class NeuralNetDenseDistr {

    private static final int nNodes = 4;
    private static final int batchSize = 100;
    private static final int batchSizeLocal = batchSize / nNodes;

    /* Input data set parameters */
    private static final String[] trainDatasetFiles     = { "../data/distributed/neural_network_train_dense_1.csv",
                                                            "../data/distributed/neural_network_train_dense_2.csv",
                                                            "../data/distributed/neural_network_train_dense_3.csv",
                                                            "../data/distributed/neural_network_train_dense_4.csv" };
    private static final String[] trainGroundTruthFiles = { "../data/distributed/neural_network_train_ground_truth_1.csv",
                                                            "../data/distributed/neural_network_train_ground_truth_2.csv",
                                                            "../data/distributed/neural_network_train_ground_truth_3.csv",
                                                            "../data/distributed/neural_network_train_ground_truth_4.csv" };
    private static final String testDatasetFile      = "../data/batch/neural_network_test.csv";
    private static final String testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

    private static DistributedStep2Master net;
    private static DistributedStep1Local[] netLocal = new DistributedStep1Local[nNodes];

    private static TrainingModel[] trainingModel = new TrainingModel[nNodes];

    private static Tensor[] trainingData = new Tensor[nNodes];
    private static Tensor[] trainingGroundTruth = new Tensor[nNodes];

    private static TrainingTopology topology;
    private static TrainingTopology[] topologyLocal = new TrainingTopology[nNodes];

    private static PredictionModel predictionModel;
    private static PredictionResult predictionResult;
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        initializeNetwork();
        trainModel();
        testModel();
        printResults();

        context.dispose();
    }

    private static void initializeNetwork() throws java.io.FileNotFoundException, java.io.IOException {
        com.intel.daal.algorithms.optimization_solver.sgd.Batch sgdAlgorithm =
            new com.intel.daal.algorithms.optimization_solver.sgd.Batch(context, Float.class, com.intel.daal.algorithms.optimization_solver.sgd.Method.defaultDense);
        sgdAlgorithm.parameter.setBatchSize(batchSizeLocal);

        /* Read training data set from a .csv file and create tensors to store input data */
        for (int node = 0; node < nNodes; node++) {
            trainingData[node] = Service.readTensorFromCSV(context, trainDatasetFiles[node]);
            trainingGroundTruth[node] = Service.readTensorFromCSV(context, trainGroundTruthFiles[node], true);
        }

        /* Configure the neural network */
        net = new DistributedStep2Master(context, sgdAlgorithm);
        topology = NeuralNetConfiguratorDistr.configureNet(context);
        net.parameter.setOptimizationSolver(sgdAlgorithm);

        long[] sampleSize = trainingData[0].getDimensions();
        sampleSize[0] = batchSizeLocal;
        /* Initialize the neural network on master node */
        net.initialize(sampleSize, topology);

        for (int node = 0; node < nNodes; node++) {
            /* Configure the neural network */
            topologyLocal[node] = NeuralNetConfiguratorDistr.configureNet(context);

            netLocal[node] = new DistributedStep1Local(context);

            trainingModel[node] = new TrainingModel(context);
            trainingModel[node].initialize(Float.class, sampleSize, topologyLocal[node]);

            /* Pass a model from master node to the algorithms on local nodes */
            netLocal[node].input.set(DistributedStep1LocalInputId.inputModel, trainingModel[node]);
        }
    }

    private static void trainModel() throws java.io.FileNotFoundException, java.io.IOException {
        /* Create stochastic gradient descent (SGD) optimization solver algorithm */
        com.intel.daal.algorithms.optimization_solver.sgd.Batch sgdAlgorithm =
            new com.intel.daal.algorithms.optimization_solver.sgd.Batch(context, Float.class, com.intel.daal.algorithms.optimization_solver.sgd.Method.defaultDense);

        /* Set learning rate for the optimization solver used in the neural network */
        double[] learningRateArray = new double[1];
        learningRateArray[0] = 0.001;
        sgdAlgorithm.parameter.setLearningRateSequence(new HomogenNumericTable(context, learningRateArray, 1, 1));
        sgdAlgorithm.parameter.setBatchSize(batchSizeLocal);

        /* Set the optimization solver for the neural network training */
        net.parameter.setOptimizationSolver(sgdAlgorithm);

        /* Run the neural network training */
        int nSamples = (int)trainingData[0].getDimensions()[0];
        for (int i = 0; i < nSamples - batchSizeLocal + 1; i += batchSizeLocal) {
            /* Compute weights and biases for the batch of inputs on local nodes */
            for (int node = 0; node < nNodes; node++) {
                /* Pass a training data set and dependent values to the algorithm */
                netLocal[node].input.set(TrainingInputId.data, Service.getNextSubtensor(context, trainingData[node], i, batchSizeLocal));
                netLocal[node].input.set(TrainingInputId.groundTruth, Service.getNextSubtensor(context, trainingGroundTruth[node], i, batchSizeLocal));

                /* Compute weights and biases on local node */
                PartialResult partialResult = netLocal[node].compute();

                /* Pass computed weights and biases to the master algorithm */
                net.input.add(DistributedStep2MasterInputId.partialResults, node, partialResult);
            }

            /* Update weights and biases on master node */
            DistributedPartialResult result = net.compute();
            NumericTable wb = result.get(DistributedPartialResultId.resultFromMaster).get(TrainingResultId.model).getWeightsAndBiases();

            /* Update weights and biases on local nodes */
            for (int node = 0; node < nNodes; node++) {
                netLocal[node].input.get(DistributedStep1LocalInputId.inputModel).setWeightsAndBiases(wb);
            }
        }

        /* Finalize neural network training on the master node */
        TrainingResult result = net.finalizeCompute();

        /* Retrieve training and prediction models of the neural network */
        TrainingModel trainingModel = result.get(TrainingResultId.model);
        predictionModel = trainingModel.getPredictionModel(Float.class);
    }

    private static void testModel() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read testing data set from a .csv file and create a tensor to store input data */
        Tensor predictionData = Service.readTensorFromCSV(context, testDatasetFile);

        /* Create an algorithm to compute the neural network predictions */
        PredictionBatch net = new PredictionBatch(context);

        long[] predictionDimensions = predictionData.getDimensions();
        net.parameter.setBatchSize(predictionDimensions[0]);

        /* Set input objects for the prediction neural network */
        net.input.set(PredictionTensorInputId.data, predictionData);
        net.input.set(PredictionModelInputId.model, predictionModel);

        /* Run the neural network prediction */
        predictionResult = net.compute();
    }

    private static void printResults() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read testing ground truth from a .csv file and create a tensor to store the data */
        Tensor predictionGroundTruth = Service.readTensorFromCSV(context, testGroundTruthFile);

        /* Print results of the neural network prediction */
        Service.printTensors("Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):",
                             predictionGroundTruth, predictionResult.get(PredictionResultId.prediction), 20);
    }
}
