/* file: neural_net_dense_distributed_mpi.cpp */
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
!    C++ example of neural network training and scoring in the distributed processing mode
!******************************************************************************/

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "neural_net_dense_distributed_mpi.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::services;

typedef std::vector<byte> ByteBuffer;

/* Input data set parameters */
const string trainDatasetFileNames[4] =
{
    "./data/distributed/neural_network_train_dense_1.csv", "./data/distributed/neural_network_train_dense_2.csv",
    "./data/distributed/neural_network_train_dense_3.csv", "./data/distributed/neural_network_train_dense_4.csv"
};
const string trainGroundTruthFileNames[4] =
{
    "./data/distributed/neural_network_train_ground_truth_1.csv", "./data/distributed/neural_network_train_ground_truth_2.csv",
    "./data/distributed/neural_network_train_ground_truth_3.csv", "./data/distributed/neural_network_train_ground_truth_4.csv"
};
string testDatasetFile     = "./data/distributed/neural_network_test.csv";
string testGroundTruthFile = "./data/distributed/neural_network_test_ground_truth.csv";

const size_t nNodes = 4;
const size_t batchSize = 100;
const size_t batchSizeLocal = batchSize / nNodes;

TensorPtr trainingData;
TensorPtr trainingGroundTruth;
prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;
training::TopologyPtr topology;

/* Algorithms to train neural network */
SharedPtr<training::Distributed<step1Local> > netLocal;
SharedPtr<training::Distributed<step2Master> > netMaster;

int rankId, comm_size;
#define mpi_root 0

void initializeNetwork();
void trainModel();
void testModel();
void printResults();

static NumericTablePtr broadcastWeightsAndBiasesToNodes(
    NumericTable *wb, size_t &wbArchLength, ByteBuffer &wbBuffer);

static void gatherPartialResultsFromNodes(const training::PartialResultPtr &partialResult,
        training::PartialResultPtr partialResults[],
        size_t &partialResultArchLength,
        ByteBuffer &partialResultLocalBuffer,
        ByteBuffer &partialResultMasterBuffer);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);

    initializeNetwork();
    trainModel();

    if(rankId == mpi_root)
    {
        testModel();
        printResults();
    }

    MPI_Finalize();

    return 0;
}

void initializeNetwork()
{
    /* Read training data set from a .csv file and create tensors to store input data */
    trainingData = readTensorFromCSV(trainDatasetFileNames[rankId]);
    trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId], true);

    Collection<size_t> sampleSize = trainingData->getDimensions();
    sampleSize[0] = batchSizeLocal;

    /* Configure the neural network topology */
    topology = configureNet();

    /* Create AdaGrad optimization solver algorithm */
    SharedPtr<optimization_solver::adagrad::Batch<> > solver(new optimization_solver::adagrad::Batch<>());

    /* Set learning rate for the optimization solver used in the neural network */
    float learningRate = 0.001f;
    solver->parameter.learningRate = NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, learningRate));
    solver->parameter.batchSize = batchSizeLocal;
    solver->parameter.optionalResultRequired = true;

    training::ModelPtr trainingModel;
    if (rankId == mpi_root)
    {
        /* Set the optimization solver for the neural network training */
        netMaster = SharedPtr<training::Distributed<step2Master> >(new training::Distributed<step2Master>(solver));

        /* Initialize the neural network on master node */
        netMaster->initialize(sampleSize, *topology);

        trainingModel = netMaster->getResult()->get(training::model);
    }
    else
    {
        trainingModel = training::ModelPtr(new training::Model());
        trainingModel->initialize<float>(sampleSize, *topology);
    }

    netLocal = SharedPtr<training::Distributed<step1Local> >(new training::Distributed<step1Local>());
    /* Pass a model from master node to the algorithms on local nodes */
    netLocal->input.set(training::inputModel, trainingModel);
}

void trainModel()
{
    ByteBuffer wbBuffer(0);
    size_t wbArchLength = 0;

    size_t partialResultArchLength = 0;
    ByteBuffer partialResultLocalBuffer (0);
    ByteBuffer partialResultMasterBuffer(0);

    /* Run the neural network training */
    const size_t nSamples = trainingData->getDimensionSize(0);
    for (size_t i = 0; i < nSamples - batchSizeLocal + 1; i += batchSizeLocal)
    {
        /* Compute weights and biases for the batch of inputs on local nodes */

        /* Pass a training data set and dependent values to the algorithm */
        netLocal->input.set(training::data,        getNextSubtensor(trainingData,        i, batchSizeLocal));
        netLocal->input.set(training::groundTruth, getNextSubtensor(trainingGroundTruth, i, batchSizeLocal));

        /* Compute weights and biases derivatives on local node */
        netLocal->compute();

        training::PartialResultPtr partialResults[nNodes];

        gatherPartialResultsFromNodes(netLocal->getPartialResult(), partialResults, partialResultArchLength,
                                      partialResultLocalBuffer, partialResultMasterBuffer);

        NumericTablePtr wb;
        if(rankId == mpi_root)
        {
            for (size_t node = 0; node < nNodes; node++)
            {
                /* Pass computed weights and biases derivatives to the master algorithm */
                netMaster->input.add(training::partialResults, node, partialResults[node]);
            }

            /* Update weights and biases on master node */
            netMaster->compute();
            training::ModelPtr wbModel = netMaster->getPartialResult()->get(training::resultFromMaster)->get(training::model);
            checkPtr((void *)wbModel.get());
            wb = wbModel->getWeightsAndBiases();
        }

        /* Broadcast updated weights and biases to nodes */
        NumericTablePtr wbLocal = broadcastWeightsAndBiasesToNodes(wb.get(), wbArchLength, wbBuffer);

        netLocal->input.get(training::inputModel)->setWeightsAndBiases(wbLocal);
    }

    if(rankId == mpi_root)
    {
        /* Finalize neural network training on the master node */
        netMaster->finalizeCompute();

        /* Retrieve training and prediction models of the neural network */
        training::ModelPtr trModel = netMaster->getResult()->get(training::model);
        checkPtr((void *)trModel.get());
        predictionModel = trModel->getPredictionModel<float>();
    }
}

void testModel()
{
    /* Read testing data set from a .csv file and create a tensor to store input data */
    TensorPtr predictionData = readTensorFromCSV(testDatasetFile);

    /* Create an algorithm to compute the neural network predictions */
    prediction::Batch<> net;

    /* Set the batch size for the neural network prediction */
    net.parameter.batchSize = predictionData->getDimensionSize(0);

    /* Set input objects for the prediction neural network */
    net.input.set(prediction::model, predictionModel);
    net.input.set(prediction::data, predictionData);

    /* Run the neural network prediction */
    net.compute();

    /* Print results of the neural network prediction */
    predictionResult = net.getResult();
}

void printResults()
{
    /* Read testing ground truth from a .csv file and create a tensor to store the data */
    TensorPtr predictionGroundTruth = readTensorFromCSV(testGroundTruthFile);

    printTensors<int, float>(predictionGroundTruth, predictionResult->get(prediction::prediction),
                             "Ground truth", "Neural network predictions: each class probability",
                             "Neural network classification results (first 20 observations):", 20);
}

void gatherPartialResultsFromNodes(const training::PartialResultPtr &partialResult,
                                   training::PartialResultPtr partialResults[],
                                   size_t &partialResultArchLength,
                                   ByteBuffer &partialResultLocalBuffer,
                                   ByteBuffer &partialResultMasterBuffer)
{
    InputDataArchive dataArch;
    partialResult->serialize( dataArch );
    if (partialResultArchLength == 0)
    {
        partialResultArchLength = dataArch.getSizeOfArchive();
    }

    /* Serialized data is of equal size on each node */
    if (rankId == mpi_root && partialResultMasterBuffer.empty())
    {
        partialResultMasterBuffer.resize(partialResultArchLength * nNodes);
    }

    if (partialResultLocalBuffer.empty())
    {
        partialResultLocalBuffer.resize(partialResultArchLength);
    }
    dataArch.copyArchiveToArray(&partialResultLocalBuffer[0], partialResultArchLength);

    /* Transfer partial results to step 2 on the root node */
    MPI_Gather(&partialResultLocalBuffer[0], partialResultArchLength, MPI_BYTE, &partialResultMasterBuffer[0], partialResultArchLength, MPI_BYTE, mpi_root,
               MPI_COMM_WORLD);

    if (rankId == mpi_root)
    {
        for (size_t node = 0; node < nNodes; node++)
        {
            /* Deserialize partial results from step 1 */
            OutputDataArchive dataArch(&partialResultMasterBuffer[0] + partialResultArchLength * node, partialResultArchLength);

            partialResults[node].reset(new training::PartialResult());
            partialResults[node]->deserialize(dataArch);
        }
    }
}

NumericTablePtr broadcastWeightsAndBiasesToNodes(
    NumericTable *wb, size_t &wbArchLength, ByteBuffer &wbBuffer)
{
    /* Serialize weights and biases on the root node */
    if (rankId == mpi_root)
    {
        if (!wb)
        {
            /* Weights and biases table should be valid and not NULL on master */
            return NumericTablePtr();
        }
        InputDataArchive wbDataArch;
        wb->serialize(wbDataArch);
        if (wbArchLength == 0)
        {
            wbArchLength = wbDataArch.getSizeOfArchive();
            wbBuffer.resize(wbArchLength);
        }
        wbDataArch.copyArchiveToArray(&wbBuffer[0], wbArchLength);
    }

    MPI_Bcast(&wbArchLength, sizeof(size_t), MPI_BYTE, mpi_root, MPI_COMM_WORLD);

    if (wbBuffer.empty())
    {
        wbBuffer.resize(wbArchLength);
    }

    /* Broadcast the serialized weights and biases */
    MPI_Bcast(&wbBuffer[0], wbArchLength, MPI_BYTE, mpi_root, MPI_COMM_WORLD);

    /* Deserialize weights and biases */
    OutputDataArchive wbDataArchLocal(&wbBuffer[0], wbArchLength);

    NumericTablePtr wbLocal = NumericTablePtr(new HomogenNumericTable<>());

    wbLocal->deserialize(wbDataArchLocal);
    return wbLocal;
}
