/* file: neural_net_dense_allgather_distributed_mpi.cpp */
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
!******************************************************************************/

#include <mpi.h>
#include "daal.h"
#include "service.h"
#include "neural_net_dense_distributed_mpi.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::optimization_solver;
using namespace daal::services;

typedef optimization_solver::precomputed::Batch<> ObjFunction;

const size_t nNodes   = 4;
const float invNNodes = 1.0f / nNodes;

/* Input data set parameters */
const string trainDatasetFileNames[nNodes] =
{
    "./data/distributed/neural_network_train_dense_1.csv",
    "./data/distributed/neural_network_train_dense_2.csv",
    "./data/distributed/neural_network_train_dense_3.csv",
    "./data/distributed/neural_network_train_dense_4.csv"
};
const string trainGroundTruthFileNames[nNodes] =
{
    "./data/distributed/neural_network_train_ground_truth_1.csv",
    "./data/distributed/neural_network_train_ground_truth_2.csv",
    "./data/distributed/neural_network_train_ground_truth_3.csv",
    "./data/distributed/neural_network_train_ground_truth_4.csv"
};
string testDatasetFile     = "./data/distributed/neural_network_test.csv";
string testGroundTruthFile = "./data/distributed/neural_network_test_ground_truth.csv";

const size_t batchSizeLocal = 25;

const size_t nLayers = 4;

TensorPtr trainingData;
TensorPtr trainingGroundTruth;
training::ModelPtr trainingModel;
prediction::ResultPtr predictionResult;

int rankId, comm_size;
#define mpi_root 0

void initializeNetwork();
void trainModel();
void testModel();
void printResults();

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
    trainingData = readTensorFromCSV(trainDatasetFileNames[rankId]);
    trainingGroundTruth = readTensorFromCSV(trainGroundTruthFileNames[rankId], true);

    Collection<size_t> sampleSize = trainingData->getDimensions();
    sampleSize[0] = batchSizeLocal;

    SharedPtr<optimization_solver::adagrad::Batch<> > solver(new optimization_solver::adagrad::Batch<>());
    solver->parameter.batchSize = batchSizeLocal;

    /* Configure the neural network model on worker nodes */
    training::Parameter parameter(staticPointerCast<optimization_solver::iterative_solver::Batch, optimization_solver::adagrad::Batch<> >(solver));

    training::TopologyPtr topology = configureNet();
    trainingModel.reset(new training::Model());
    trainingModel->initialize<float>(sampleSize, *topology, parameter);
}

void initializeOptimizationSolver(optimization_solver::sgd::Batch<> &solver,
    SharedPtr<ObjFunction> &objFunction, const TensorPtr &tensor)
{
    if (tensor && tensor->getSize())
    {
        NumericTablePtr inputArgTable = createTableFromTensor(*tensor);

        solver.input.set(iterative_solver::inputArgument,  inputArgTable);

        iterative_solver::ResultPtr solverResult = solver.getResult();
        solverResult->set(iterative_solver::minimum, inputArgTable);

        NumericTablePtr derivTable(new HomogenNumericTable<>(1, tensor->getSize(), NumericTable::doAllocate));

        objFunction.reset(new ObjFunction());
        objective_function::ResultPtr result = objFunction->getResult();
        result->set(objective_function::gradientIdx, derivTable);
        objFunction->setResult(result);

        /* Set learning rate for the optimization solver used in the neural network */
        (*(HomogenNumericTable<double>::cast(solver.parameter.learningRateSequence)))[0][0] = 0.001;

        solver.parameter.function = objFunction;
        solver.parameter.nIterations = 1;
    }
}

void setNextBatchToModel(training::Model &model, const TensorPtr &dataBatch, const TensorPtr &groundTruthBatch)
{
    layers::forward::LayerIface *firstFwdLayer = model.getForwardLayer(0).get();
    layers::forward::Input *firstFwdLayerInput = firstFwdLayer->getLayerInput();
    firstFwdLayerInput->set(layers::forward::data, dataBatch);
    firstFwdLayer->getLayerResult()->setResultForBackward(firstFwdLayerInput);

    layers::forward::LayerIface *lastFwdLayer = model.getForwardLayer(nLayers-1).get();
    loss::forward::Input *lossInput = static_cast<loss::forward::Input *>(lastFwdLayer->getLayerInput());
    lossInput->set(layers::loss::forward::groundTruth, groundTruthBatch);
    lastFwdLayer->getLayerResult()->setResultForBackward(lossInput);
}

void updateParameters(optimization_solver::sgd::Batch<> &solver,
            TensorPtr &tensor, ByteBuffer &buffer, size_t archLength, MPI_Request &request)
{
    if (archLength > 0 && request != MPI_REQUEST_NULL)
    {
        /* Wait for partial derivatives from all nodes */
        MPI_Wait(&request, MPI_STATUS_IGNORE);

        /* Compute the sum of derivatives received from all nodes */
        SharedPtr<HomogenNumericTable<> > sumTable = HomogenNumericTable<>::cast(
                solver.parameter.function->getResult()->get(objective_function::gradientIdx));
        float *sumData = sumTable->getArray();
        size_t sumSize = sumTable->getNumberOfRows();
        for (size_t i = 0; i < sumSize; i++)
        {
            sumData[i] = 0.0f;
        }
        for (size_t node = 0; node < nNodes; node++)
        {
            /* Retrieve a partial derivative from a node */
            TensorPtr tensor = Tensor::cast(deserializeDAALObject(&buffer[node * archLength], archLength));
            SubtensorDescriptor<float> subtensor;
            tensor->getSubtensor(0, 0, 0, tensor->getDimensionSize(0), readOnly, subtensor);
            const float *data = subtensor.getPtr();
            for (size_t i = 0; i < sumSize; i++)
            {
                sumData[i] += data[i];
            }
            tensor->releaseSubtensor(subtensor);
        }
        /* Compute the average of derivatives received from all nodes */
        for (size_t i = 0; i < sumSize; i++)
        {
            sumData[i] *= invNNodes;
        }

        /* Update weights on all nodes by performing a step of optimization algorithm */
        solver.compute();

        NumericTable *minimumTable = solver.getResult()->get(iterative_solver::minimum).get();
        copyTableToTensor(*minimumTable, *tensor);
    }
}

void allgatherDerivatives(Tensor *tensor, ByteBuffer &buffer, ByteBuffer &bufferLocal, MPI_Request &request)
{
    if (tensor && tensor->getSize() > 0)
    {
        serializeDAALObject(tensor, bufferLocal);
        size_t bufSize = bufferLocal.size();
        if (buffer.size() == 0) { buffer.resize(bufferLocal.size() * nNodes); }

        /* Initiate asynchronous transfer of partial derivatives to all nodes */
        MPI_Iallgather(&bufferLocal[0], bufSize, MPI_BYTE, &buffer[0], bufSize, MPI_BYTE, MPI_COMM_WORLD, &request);
    }
}

void trainModel()
{
    TensorPtr wTensor[nLayers];                                 // tensors of weights of each layer
    TensorPtr bTensor[nLayers];                                 // tensors of biases of each layer
    SharedPtr<ObjFunction> wObjFunc[nLayers];                   // objective functions associated with weight derivatives
    SharedPtr<ObjFunction> bObjFunc[nLayers];                   // objective functions associated with bias derivatives
    optimization_solver::sgd::Batch<> wSolver[nLayers];    // optimization solvers associated with weights of each layer
    optimization_solver::sgd::Batch<> bSolver[nLayers];    // optimization solvers associated with biases of each layer

    /* Set input arguments for the optimization solvers */
    for (size_t l = 0; l < nLayers; l++)
    {
        layers::forward::Input *fwdInput = trainingModel->getForwardLayer(l)->getLayerInput();
        wTensor[l] = fwdInput->get(layers::forward::weights);
        bTensor[l] = fwdInput->get(layers::forward::biases);
        initializeOptimizationSolver(wSolver[l], wObjFunc[l], wTensor[l]);
        initializeOptimizationSolver(bSolver[l], bObjFunc[l], bTensor[l]);
    }

    ByteBuffer wDerBuffersLocal[nLayers];   // buffer for serialized weight derivatives on local node
    ByteBuffer bDerBuffersLocal[nLayers];   // buffer for serialized bias derivatives on local node
    ByteBuffer wDerBuffers[nLayers];        // buffer for serialized weight derivatives from all nodes
    ByteBuffer bDerBuffers[nLayers];        // buffer for serialized bias derivatives from all nodes

    std::vector<MPI_Request> wDerRequests(nLayers, MPI_REQUEST_NULL);
    std::vector<MPI_Request> bDerRequests(nLayers, MPI_REQUEST_NULL);

    const size_t nSamples = trainingData->getDimensionSize(0);

    for (size_t i = 0; i < nSamples; i += batchSizeLocal)
    {
        /* Pass a training data set and dependent values to the algorithm */
        setNextBatchToModel(*trainingModel, getNextSubtensor(trainingData, i, batchSizeLocal),
                                            getNextSubtensor(trainingGroundTruth, i, batchSizeLocal));
        /* FORWARD PASS */
        for (size_t l = 0; l < nLayers; l++)
        {
            /* Wait for updated derivatives from all nodes and update weights on local node
               using the optimization solver algorithm */
            updateParameters(wSolver[l], wTensor[l], wDerBuffers[l], wDerBuffersLocal[l].size(), wDerRequests[l]);
            updateParameters(bSolver[l], bTensor[l], bDerBuffers[l], bDerBuffersLocal[l].size(), bDerRequests[l]);

            /* Compute forward layer results */
            trainingModel->getForwardLayer(l)->compute();
        }

        /* BACKWARD PASS */
        for (int l = nLayers - 1; l >= 0; l--)
        {
            /* Compute weight and bias derivatives for the batch of inputs on worker nodes */
            layers::backward::LayerIfacePtr layer = trainingModel->getBackwardLayer(l);
            layer->compute();

            /* Start derivatives gathering on all nodes */
            layers::backward::ResultPtr layerResult = layer->getLayerResult();
            TensorPtr wDerTensor = layerResult->get(layers::backward::weightDerivatives);
            TensorPtr bDerTensor = layerResult->get(layers::backward::biasDerivatives);

            allgatherDerivatives(wDerTensor.get(), wDerBuffers[l], wDerBuffersLocal[l], wDerRequests[l]);
            allgatherDerivatives(bDerTensor.get(), bDerBuffers[l], bDerBuffersLocal[l], bDerRequests[l]);
        }
    }

    for (int l = nLayers - 1; l >= 0; l--)
    {
        /* Wait for updated derivatives from all nodes and update weights and biases on local node
           using the optimization solver algorithm */
        updateParameters(wSolver[l], wTensor[l], wDerBuffers[l], wDerBuffersLocal[l].size(), wDerRequests[l]);
        updateParameters(bSolver[l], bTensor[l], bDerBuffers[l], bDerBuffersLocal[l].size(), bDerRequests[l]);
    }
}

void testModel()
{
    /* Retrieve prediction model of the neural network */
    prediction::ModelPtr predictionModel = trainingModel->getPredictionModel<float>();

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
