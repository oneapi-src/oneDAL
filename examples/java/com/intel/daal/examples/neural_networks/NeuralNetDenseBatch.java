/* file: NeuralNetDenseBatch.java */
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
 //     Java example of neural network in the batch processing mode
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
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKBATCH">
 * @example NeuralNetDenseBatch.java
 */
class NeuralNetDenseBatch {

    /* Input data set parameters */
    private static final String trainDatasetFile     = "../data/batch/neural_network_train.csv";
    private static final String trainGroundTruthFile = "../data/batch/neural_network_train_ground_truth.csv";
    private static final String testDatasetFile      = "../data/batch/neural_network_test.csv";
    private static final String testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

    private static PredictionModel predictionModel;
    private static PredictionResult predictionResult;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        trainModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void trainModel() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read training data set from a .csv file and create a tensor to store input data */
        Tensor trainingData = Service.readTensorFromCSV(context, trainDatasetFile);
        Tensor trainingGroundTruth = Service.readTensorFromCSV(context, trainGroundTruthFile);

        /* Create an algorithm to compute neural network results using default method */
        TrainingBatch net = new TrainingBatch(context, Float.class, TrainingMethod.defaultDense);

        /* Set the batch size for the neural network training */
        net.parameter.setBatchSize(10);

        /* Configure the neural network */
        TrainingTopology topology = NeuralNetConfigurator.configureNet(context);
        net.initialize(trainingData.getDimensions(), topology);

        /* Set input objects for the neural network */
        net.input.set(TrainingInputId.data, trainingData);
        net.input.set(TrainingInputId.groundTruth, trainingGroundTruth);

        /* Set learning rate for the optimization solver used in the neural network */
        double[] learningRateArray = new double[1];
        learningRateArray[0] = 0.001;
        com.intel.daal.algorithms.optimization_solver.sgd.Batch sgdAlgorithm =
            new com.intel.daal.algorithms.optimization_solver.sgd.Batch(context, Double.class, com.intel.daal.algorithms.optimization_solver.sgd.Method.defaultDense);
        sgdAlgorithm.parameter.setLearningRateSequence(new HomogenNumericTable(context, learningRateArray, 1, 1));

        net.parameter.setOptimizationSolver(sgdAlgorithm);


        /* Run the neural network training */
        TrainingResult result = net.compute();

        /* Get training and prediction models of the neural network */
        TrainingModel trainingModel = result.get(TrainingResultId.model);
        predictionModel = trainingModel.getPredictionModel(Float.class);
    }

    private static void testModel() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read testing data set from a .csv file and create a tensor to store input data */
        Tensor predictionData = Service.readTensorFromCSV(context, testDatasetFile);

        /* Create an algorithm to compute the neural network predictions */
        PredictionBatch net = new PredictionBatch(context, Float.class, PredictionMethod.defaultDense);

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
