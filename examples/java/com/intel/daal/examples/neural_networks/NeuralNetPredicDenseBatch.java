/* file: NeuralNetPredicDenseBatch.java */
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
 //     Java example of neural network in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.neural_networks;

import com.intel.daal.algorithms.neural_networks.*;
import com.intel.daal.algorithms.neural_networks.prediction.*;
import com.intel.daal.algorithms.neural_networks.training.*;
import com.intel.daal.algorithms.neural_networks.layers.Parameter;
import com.intel.daal.algorithms.neural_networks.layers.LayerDescriptor;
import com.intel.daal.algorithms.neural_networks.layers.NextLayers;
import com.intel.daal.algorithms.neural_networks.layers.ForwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.BackwardLayer;
import com.intel.daal.algorithms.neural_networks.layers.ForwardInput;
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
 * <a name="DAAL-EXAMPLE-JAVA-NEURALNETWORKPREDICTIONBATCH">
 * @example NeuralNetPredicDenseBatch.java
 */
class NeuralNetPredicDenseBatch {

    /* Input data set parameters */
    private static final String testDatasetFile      = "../data/batch/neural_network_test.csv";
    private static final String testGroundTruthFile  = "../data/batch/neural_network_test_ground_truth.csv";

    /* Weights and biases obtained on the training stage */
    private static final String fc1WeightsFile      = "../data/batch/fc1_weights.csv";
    private static final String fc1BiasesFile       = "../data/batch/fc1_biases.csv";
    private static final String fc2WeightsFile      = "../data/batch/fc2_weights.csv";
    private static final String fc2BiasesFile       = "../data/batch/fc2_biases.csv";

    private static Tensor predictionData;
    private static PredictionModel predictionModel;
    private static PredictionResult predictionResult;

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        createModel();

        testModel();

        printResults();

        context.dispose();
    }

    private static void createModel() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read testing data set from a .csv file and create a tensor to store input data */
        predictionData = Service.readTensorFromCSV(context, testDatasetFile);

        /* Retrieve training and prediction models of the neural network */
        PredictionTopology topology = NeuralNetPredicConfigurator.configureNet(context);
        predictionModel = new PredictionModel(context, topology);

        /* Read 1st fully-connected layer weights and biases from CSV file */
        /* 1st fully-connected layer weights are a 2D tensor of size 5 x 20 */
        Tensor fc1Weights = Service.readTensorFromCSV(context, fc1WeightsFile);
        /* 1st fully-connected layer biases are a 1D tensor of size 5 */
        Tensor fc1Biases = Service.readTensorFromCSV(context, fc1BiasesFile);

        /* Set weights and biases of the 1st fully-connected layer */
        ForwardInput fc1Input = predictionModel.getLayer(0).getLayerInput();
        fc1Input.set(ForwardInputId.weights, fc1Weights);
        fc1Input.set(ForwardInputId.biases, fc1Biases);

        /* Set flag that specifies that weights and biases of the 1st fully-connected layer are initialized */
        predictionModel.getLayer(0).getLayerParameter().setWeightsAndBiasesInitializationFlag(true);

        /* Read 2nd fully-connected layer weights and biases from CSV file */
        /* 2nd fully-connected layer weights are a 2D tensor of size 2 x 5 */
        Tensor fc2Weights = Service.readTensorFromCSV(context, fc2WeightsFile);
        /* 2nd fully-connected layer biases are a 1D tensor of size 2 */
        Tensor fc2Biases = Service.readTensorFromCSV(context, fc2BiasesFile);

        /* Set weights and biases of the 2nd fully-connected layer */
        ForwardInput fc2Input = predictionModel.getLayer(1).getLayerInput();
        fc2Input.set(ForwardInputId.weights, fc2Weights);
        fc2Input.set(ForwardInputId.biases, fc2Biases);

        /* Set flag that specifies that weights and biases of the 2nd fully-connected layer are initialized */
        predictionModel.getLayer(1).getLayerParameter().setWeightsAndBiasesInitializationFlag(true);
    }

    private static void testModel() {

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
