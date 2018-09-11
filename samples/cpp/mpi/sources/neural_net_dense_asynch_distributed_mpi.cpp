/* file: neural_net_dense_asynch_distributed_mpi.cpp */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
!    using asynchronous communications
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
typedef std::vector<MPI_Request> RequestBuffer;


const size_t nWorkers = 3;
const size_t nNodes   = nWorkers + 1;
const size_t nPartialResultsToUpdateWeights = nWorkers;

/* Input data set parameters */
const string trainDatasetFileNames[nWorkers] =
{
    "./data/distributed/neural_network_train_dense_1.csv",
    "./data/distributed/neural_network_train_dense_2.csv",
    "./data/distributed/neural_network_train_dense_3.csv"
};
const string trainGroundTruthFileNames[nWorkers] =
{
    "./data/distributed/neural_network_train_ground_truth_1.csv",
    "./data/distributed/neural_network_train_ground_truth_2.csv",
    "./data/distributed/neural_network_train_ground_truth_3.csv"
};
string testDatasetFile     = "./data/distributed/neural_network_test.csv";
string testGroundTruthFile = "./data/distributed/neural_network_test_ground_truth.csv";

const size_t batchSizeLocal = 25;
const size_t nIterations = 60;

TensorPtr trainingData;
TensorPtr trainingGroundTruth;
prediction::ModelPtr predictionModel;
prediction::ResultPtr predictionResult;
training::TopologyPtr topology;
training::TopologyPtr topologyMaster;

/* Algorithms to train neural network */
SharedPtr<training::Distributed<step1Local> > netLocal;
SharedPtr<training::Distributed<step2Master> > netMaster;

int rankId, comm_size;
#define mpi_root 0

const int partialResultTag       = 0;
const int partialResultLengthTag = 1;
const int wbTag                  = 2;

void initializeNetwork();
void trainModel();
void testModel();
void printResults();


NumericTablePtr pullWeightsAndBiasesFromMaster(size_t &wbArchLength, ByteBuffer &wbBuffer, MPI_Request &wbRequest);

static void sendPartialResultToMaster(const training::PartialResultPtr &partialResult,
                                      size_t &partialResultArchLength,
                                      ByteBuffer &partialResultBuffer,
                                      MPI_Request &prRequest);

static training::PartialResultPtr deserializePartialResultFromNode(
    size_t &partialResultArchLength, byte *partialResultBuffer);

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
    if (rankId == mpi_root)
    {
        trainingData = readTensorFromCSV(trainDatasetFileNames[0]);
        trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[0], true);
    }
    else
    {
        trainingData = readTensorFromCSV(trainDatasetFileNames[rankId - 1]);
        trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId - 1], true);
    }

    /* Initialize the neural network on master node */
    services::Collection<size_t> sampleSize = trainingData->getDimensions();
    sampleSize[0] = batchSizeLocal;

    /* Configure the neural network topology */
    topology = configureNet();

    training::ModelPtr trainingModel;
    if (rankId == mpi_root)
    {
        /* Create AdaGrad optimization solver algorithm */
        SharedPtr<optimization_solver::adagrad::Batch<> > solver(new optimization_solver::adagrad::Batch<>());

        /* Set learning rate for the optimization solver used in the neural network */
        float learningRate = 0.001f;
        solver->parameter.learningRate = NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, learningRate));
        solver->parameter.batchSize = batchSizeLocal;
        solver->parameter.optionalResultRequired = true;

        /* Set the optimization solver for the neural network training */
        netMaster = SharedPtr<training::Distributed<step2Master> >(new training::Distributed<step2Master>(solver));

        /* Initialize the neural network on master node */
        netMaster->initialize(sampleSize, *topology);

        trainingModel = netMaster->getResult()->get(training::model);
    }
    else
    {
        training::ModelPtr trainingModel(new training::Model());
        trainingModel->initialize<float>(sampleSize, *topology);

        /* Set the batch size for the neural network training */
        netLocal = SharedPtr<training::Distributed<step1Local> >(new training::Distributed<step1Local>());
        netLocal->input.set(training::inputModel, trainingModel);
    }
}

void trainModel()
{
    ByteBuffer wbBuffer(0);             // buffer for serialized weights and biases
    size_t wbArchLength = 0;            // length of the buffer for serialized weights and biases

    ByteBuffer partialResultBuffer(0);  // buffer for serialized partial results (derivatives)
    size_t partialResultArchLength = 0; // length of the buffer for serialized partial results

    /* Run the neural network training on worker node */
    const size_t nSamples = trainingData->getDimensionSize(0);

    if (rankId == mpi_root)
    {
        /* Serialize weights and biases on master node */
        training::ModelPtr wbModel = netMaster->getPartialResult()->get(training::resultFromMaster)->get(training::model);
        checkPtr((void *)wbModel.get());
        NumericTablePtr wb = wbModel->getWeightsAndBiases();
        InputDataArchive wbDataArch;
        wb->serialize(wbDataArch);

        wbArchLength = wbDataArch.getSizeOfArchive();

        wbBuffer.resize(wbArchLength);
        wbDataArch.copyArchiveToArray(&wbBuffer[0], wbArchLength);
    }

    /* Broadcast the length of the buffer for serialized weights and biases */
    MPI_Bcast(&wbArchLength, sizeof(size_t), MPI_BYTE, mpi_root, MPI_COMM_WORLD);

    if (rankId != mpi_root)
    {
        /* Process input data on worker nodes */
        wbBuffer.resize(wbArchLength);

        MPI_Request prRequest;
        MPI_Request wbRequest;

        for (size_t i = 0; i < nSamples; i += batchSizeLocal)
        {
            /* Compute weights and biases for the batch of inputs on worker nodes */

            /* Pass a training data set and dependent values to the algorithm */
            netLocal->input.set(training::data,        getNextSubtensor(trainingData,        i, batchSizeLocal));
            netLocal->input.set(training::groundTruth, getNextSubtensor(trainingGroundTruth, i, batchSizeLocal));

            if (i > 0)
            {
                /* Pull the updated weights and biases from master node */
                NumericTablePtr wbLocal = pullWeightsAndBiasesFromMaster(wbArchLength, wbBuffer, wbRequest);
                netLocal->input.get(training::inputModel)->setWeightsAndBiases(wbLocal);
            }
            if (i + batchSizeLocal < nSamples)
            {
                /* Request the updated weights and biases from master node */
                MPI_Irecv(&wbBuffer[0], wbArchLength, MPI_BYTE, mpi_root, wbTag, MPI_COMM_WORLD, &wbRequest);
            }

            /* Perform forward and backward pass through the neural network
               to compute weights and biases derivatives on worker node */
            netLocal->compute();

            /* Send the derivatives to master node */
            if (i > 0)
            {
                MPI_Wait(&prRequest, MPI_STATUS_IGNORE);
            }
            sendPartialResultToMaster(netLocal->getPartialResult(), partialResultArchLength, partialResultBuffer, prRequest);

            if (i == 0)
            {
                MPI_Send(&partialResultArchLength, sizeof(size_t), MPI_BYTE, mpi_root,
                         partialResultLengthTag, MPI_COMM_WORLD);
            }
        }
        MPI_Wait(&prRequest, MPI_STATUS_IGNORE);
    }
    else
    {
        MPI_Request prRequests[nPartialResultsToUpdateWeights];
        MPI_Status prStatuses[nPartialResultsToUpdateWeights];

        MPI_Request wbRequests[nWorkers];
        MPI_Status wbStatuses[nWorkers];

        ByteBuffer partialResultMasterBuffer(0);
        {
            /* Receive the length of the buffer for serialized partial results */
            MPI_Request request;
            MPI_Irecv(&partialResultArchLength, sizeof(size_t), MPI_BYTE, MPI_ANY_SOURCE,
                      partialResultLengthTag, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);
            partialResultMasterBuffer.resize(nPartialResultsToUpdateWeights * partialResultArchLength);
        }

        for (size_t epoch = 0; epoch < nIterations; epoch++)
        {
            /* Receive partial results from worker nodes */
            for (size_t i = 0; i < nPartialResultsToUpdateWeights; i++)
            {
                MPI_Irecv(&partialResultMasterBuffer[i * partialResultArchLength], partialResultArchLength, MPI_BYTE, MPI_ANY_SOURCE, partialResultTag,
                          MPI_COMM_WORLD, &prRequests[i]);
            }

            for (size_t i = 0; i < nPartialResultsToUpdateWeights; i++)
            {
                int nodeIndex;
                /* Receive partial result from worker node */
                MPI_Waitany(nPartialResultsToUpdateWeights, prRequests, &nodeIndex, MPI_STATUS_IGNORE);
                prRequests[nodeIndex] = MPI_REQUEST_NULL;

                training::PartialResultPtr partialResult = deserializePartialResultFromNode(partialResultArchLength, &partialResultMasterBuffer[nodeIndex * partialResultArchLength]);

                /* Pass computed weights and biases derivatives to the master algorithm */
                netMaster->input.add(training::partialResults, i, partialResult);
            }

            /* Perform the step of optimization algorithm using partial results from worker nodes
               to update weights and biases on master node */
            netMaster->compute();

            if (epoch > 0)
            {
                MPI_Waitall(nWorkers, wbRequests, wbStatuses);
            }

            if (epoch < nIterations - 1)
            {
                /* Send the updated weights and biases to worker nodes */
                training::ModelPtr wbModel = netMaster->getPartialResult()->get(training::resultFromMaster)->get(training::model);
                checkPtr((void *)wbModel.get());
                NumericTablePtr wb = wbModel->getWeightsAndBiases();

                /* Serialize weights and biases */
                InputDataArchive wbDataArch;
                wb->serialize(wbDataArch);
                wbDataArch.copyArchiveToArray(&wbBuffer[0], wbArchLength);

                for (size_t i = 0; i < nWorkers; i++)
                {
                    MPI_Isend(&wbBuffer[0], wbArchLength, MPI_BYTE, i + 1, wbTag, MPI_COMM_WORLD, &wbRequests[i]);
                }
            }
        }

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

void sendPartialResultToMaster(const training::PartialResultPtr &partialResult,
                               size_t &partialResultArchLength,
                               ByteBuffer &partialResultBuffer,
                               MPI_Request &prRequest)
{
    InputDataArchive dataArch;
    NumericTablePtr wbDer = partialResult->get(training::derivatives);
    partialResult->serialize( dataArch );

    if (partialResultArchLength == 0)
    {
        partialResultArchLength = dataArch.getSizeOfArchive();
        partialResultBuffer.resize(partialResultArchLength);
    }

    dataArch.copyArchiveToArray(&partialResultBuffer[0], partialResultArchLength);

    MPI_Isend(&partialResultBuffer[0], partialResultArchLength, MPI_BYTE, mpi_root, partialResultTag, MPI_COMM_WORLD, &prRequest);
}

training::PartialResultPtr deserializePartialResultFromNode(
    size_t &partialResultArchLength, byte *partialResultBuffer)
{
    /* Deserialize partial results from step 1 */
    OutputDataArchive dataArch(partialResultBuffer, partialResultArchLength);

    training::PartialResultPtr partialResult(new training::PartialResult());
    partialResult->deserialize(dataArch);
    return partialResult;
}

NumericTablePtr pullWeightsAndBiasesFromMaster(size_t &wbArchLength, ByteBuffer &wbBuffer, MPI_Request &wbRequest)
{
    MPI_Wait(&wbRequest, MPI_STATUS_IGNORE);

    /* Deserialize weights and biases */
    OutputDataArchive wbDataArchLocal(&wbBuffer[0], wbArchLength);

    NumericTablePtr wbLocal(new HomogenNumericTable<>());

    wbLocal->deserialize(wbDataArchLocal);

    return wbLocal;
}
