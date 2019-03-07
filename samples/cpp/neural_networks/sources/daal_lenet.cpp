/* file: daal_lenet.cpp */
/*******************************************************************************
* Copyright 2017-2019 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
*******************************************************************************/

/*
!  Content:
!    C++ example of neural network training and scoring with LeNet topology
!******************************************************************************/

#include "service.h"
#include "daal_lenet.h"
#include "image_dataset.h"

/* Trains AlexNet with given dataset reader */
prediction::ModelPtr train(DatasetReader *reader);

/* Tests AlexNet with given dataset reader */
float test(prediction::ModelPtr predictionModel, DatasetReader *reader);

/* Constructs the optimization solver with given learning rate */
SharedPtr<optimization_solver::adagrad::Batch<float> > getOptimizationSolver(float learningRate);

size_t trainDatasetObjectsNum = 55000;
size_t testDatasetObjectsNum  = 10000;
float top1ErrorRateThreshold  = 0.97;
const size_t maxIteraions     = 10;
const float learningRate      = 0.01;
const size_t batchSize        = 64;

const std::string defaultDatasetsPath = "./data";
const std::string datasetFileNames[] =
{
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
};

int main(int argc, char *argv[])
{
    std::string userDatasetsPath = getUserDatasetPath(argc, argv);
    std::string datasetsPath = selectDatasetPathOrExit(
        defaultDatasetsPath, userDatasetsPath, datasetFileNames, 4);

    printf("Data loading started... \n");

    /* Form path to the training and testing datasets */
    std::string trainDatasetImages = datasetsPath + "/" + datasetFileNames[0];
    std::string trainDatasetLabels = datasetsPath + "/" + datasetFileNames[1];
    std::string testDatasetImages  = datasetsPath + "/" + datasetFileNames[2];
    std::string testDatasetLabels  = datasetsPath + "/" + datasetFileNames[3];

    /* Create MNIST dataset reader and setup paths */
    DatasetReader_MNIST<float> reader;
    reader.setTrainBatch(trainDatasetImages, trainDatasetLabels, trainDatasetObjectsNum);
    reader.setTestBatch(testDatasetImages, testDatasetLabels, testDatasetObjectsNum);
    reader.read();

    printf("Data loaded \n");

    printf("LeNet training started... \n");
    prediction::ModelPtr predictionModel = train(&reader);
    printf("LeNet training completed \n");

    printf("LeNet testing started \n");
    float top1ErrorRate = test(predictionModel, &reader);
    printf("LeNet testing completed \n");

    std::cout << "Top-1 error = " << top1ErrorRate * 100 << "%" << std::endl;
    return ((1.0f - top1ErrorRate) > top1ErrorRateThreshold) ? 0 : -1;
}

prediction::ModelPtr train(DatasetReader *reader)
{
    /* Fetch LeNet topology (configureNet defined in daal_lenet.h) */
    training::TopologyPtr topology = configureNet();

    /* Create the neural network training algorithm and set batch size and optimization solver */
    training::Batch<> net(getOptimizationSolver(learningRate));

    /* Fetch the training data and labels */
    TensorPtr trainingData        = reader->getTrainData();
    TensorPtr trainingGroundTruth = reader->getTrainGroundTruth();

    /* Initialize neural network with given topology */
    Collection<size_t> sampleSize = trainingData->getDimensions();
    sampleSize[0] = batchSize;
    net.initialize(sampleSize, *topology);

    /* Set the input data batch to the neural network */
    net.input.set(training::data, trainingData);

    /* Set the input ground truth (labels) batch to the neural network */
    net.input.set(training::groundTruth, trainingGroundTruth);

    for (size_t iter = 0; iter < maxIteraions; iter++)
    {
        /* Compute the neural network forward and backward passes and update */
        /* weights and biases according to the optimization solver */
        net.compute();
    }

    /* Get prediction model */
    training::ResultPtr trainingResult = net.getResult();
    training::ModelPtr trainedModel = trainingResult->get(training::model);
    return trainedModel->getPredictionModel<float>();
}

float test(prediction::ModelPtr predictionModel, DatasetReader *reader)
{
    /* Create the neural network prediction algorithm */
    prediction::Batch<> net;

    /* Fetch the testing data and labels */
    TensorPtr testingData         = reader->getTestData();
    TensorPtr testingGroundTruth  = reader->getTestGroundTruth();

    /* Set the prediction model retrieved from the training stage */
    net.input.set(prediction::model, predictionModel);

    /* Set the input ground truth (labels) batch to the neural network */
    net.input.set(prediction::data, testingData);

    /* Compute the neural network forward pass */
    net.compute();

    TensorPtr prediction = net.getResult()->get(prediction::prediction);

    /* Print first 100 predicted classes for each test sample */
    std::cout << "First 100 predictions ( predicted | ground truth ):" << std::endl;
    printPredictedClasses(prediction, testingGroundTruth, 100);

    /* Create auxiliary object to compute error rates (defined in services.h) */
    ClassificationErrorCounter errorRateCounter(prediction, testingGroundTruth);
    return errorRateCounter.getTop1ErrorRate();
}

SharedPtr<optimization_solver::adagrad::Batch<float> > getOptimizationSolver(float learningRate)
{
    /* Create 1 x 1 NumericTable to store learning rate */
    NumericTablePtr learningRateSequence = NumericTablePtr(new HomogenNumericTable<float>(
        1, 1, NumericTable::doAllocate, learningRate));

    /* Create AdaGrad optimization solver and set learning rate */
    optimization_solver::adagrad::Batch<float> *optalg = new optimization_solver::adagrad::Batch<float>();
    optalg->parameter.learningRate = learningRateSequence;
    optalg->parameter.batchSize = batchSize;
    optalg->parameter.nIterations = trainDatasetObjectsNum / batchSize;
    return SharedPtr<optimization_solver::adagrad::Batch<float> >(optalg);
}
