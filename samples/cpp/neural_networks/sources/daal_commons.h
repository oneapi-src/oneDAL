/* file: daal_commons.h */
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
!    Common functions for traininig and testing neural networks
!******************************************************************************/

#ifndef _DAAL_COMMONS_H
#define _DAAL_COMMONS_H

#include "service.h"
#include "blob_dataset.h"

const size_t batchSize          = 1;
const size_t trainingIterations = 1;

prediction::ModelPtr trainClassifier(training::TopologyPtr topology, BlobDatasetReader *reader);

float testClassifier(prediction::ModelPtr predictionModel, BlobDatasetReader *reader);

SharedPtr<optimization_solver::sgd::Batch<float> > getDefaultOptimizationSolver(float learningRate = 0.001);

Collection<size_t> getLastLayersIndices(const training::Topology &topology);

void setGroundTruthForMultipleOutputs(training::Input &trainNetInput,
                                      const Collection<size_t> lastLayerIndices,
                                      const TensorPtr &groundTruth);


/* Trains neural network with given dataset reader */
prediction::ModelPtr trainClassifier(training::TopologyPtr topology, BlobDatasetReader *reader)
{
    std::cout << "Training started with batch size = [" << batchSize << "]" << std::endl;

    /* Get collection of last layer indices from topology */
    Collection<size_t> lastLayerIndices = getLastLayersIndices(*topology);

    /* Create the neural network training algorithm and set batch size and optimization solver */
    training::Batch<> net(getDefaultOptimizationSolver());
    net.parameter.optimizationSolver->getParameter()->nIterations = trainingIterations / reader->getTotalNumberOfObjects();

    /* Initialize neural network with given topology */
    net.initialize(reader->getBatchDimensions(), *topology);

    size_t batchCounter = 0;
    for (size_t i = 0; i < trainingIterations; i++)
    {
        /* Reset reader's iterator the dataset begining */
        reader->reset();

        /* Advance dataset reader's iterator to the next batch */
        while (reader->next())
        {
            batchCounter++;

            /* Set the input data batch to the neural network */
            net.input.set(training::data, reader->getBatch());

            /* Set the input ground truth (labels) batch to the neural network */
            setGroundTruthForMultipleOutputs(net.input, lastLayerIndices, reader->getGroundTruthBatch());

            /* Compute the neural network forward and backward passes and update */
            /* weights and biases according to the optimization solver */
            net.compute();

            std::cout << batchCounter << " train batches processed" << std::endl;
        }
    }

    /* Get prediction model */
    training::ResultPtr trainingResult = net.getResult();
    training::ModelPtr trainedModel = trainingResult->get(training::model);
    return trainedModel->getPredictionModel<float>();
}

/* Tests model with given dataset reader and return top-5 error rate */
float testClassifier(prediction::ModelPtr predictionModel, BlobDatasetReader *reader)
{
    /* Create the neural network prediction algorithm */
    prediction::Batch<> net;

    /* Set the prediction model retrieved from the training stage */
    net.input.set(prediction::model, predictionModel);

    /* Create auxiliary object to compute error rates (defined in services.h) */
    ClassificationErrorCounter errorRateCounter;

    /* Reset reader's iterator the dataset begining */
    reader->reset();

    size_t batchCounter = 0;

    /* Advance dataset reader's iterator to the next batch */
    while (reader->next())
    {
        batchCounter++;

        /* Set the input data batch to the neural network */
        net.input.set(prediction::data, reader->getBatch());

        /* Compute the neural network forward pass */
        net.compute();

        /* Get tensor of predicted probailities for each class and update error rate */
        TensorPtr prediction = net.getResult()->get(prediction::prediction);
        errorRateCounter.update(prediction, reader->getGroundTruthBatch());

        std::cout << batchCounter << " test batches processed" << std::endl;
    }

    return errorRateCounter.getTop5ErrorRate();
}

/* Constructs the optimization solver with given learning rate */
SharedPtr<optimization_solver::sgd::Batch<float> > getDefaultOptimizationSolver(float learningRate)
{
    /* Create 1 x 1 NumericTable to store learning rate */
    NumericTablePtr learningRateSequence = NumericTablePtr(new HomogenNumericTable<float>(
            1, 1, NumericTable::doAllocate, learningRate));

    /* Create SGD optimization solver and set learning rate */
    optimization_solver::sgd::Batch<float> *optalg = new optimization_solver::sgd::Batch<float>();
    optalg->parameter.learningRateSequence = learningRateSequence;
    optalg->parameter.batchSize = batchSize;
    return SharedPtr<optimization_solver::sgd::Batch<float> >(optalg);
}

Collection<size_t> getLastLayersIndices(const training::Topology &topology)
{
    Collection<size_t> lastLayerIndices;

    for (size_t i = 0; i < topology.size(); i++)
    {
        const LayerDescriptor &descriptor = topology[i];
        if (descriptor.nextLayers().size() == 0)
        {
            lastLayerIndices.push_back(descriptor.index());
        }
    }

    return lastLayerIndices;
}

void setGroundTruthForMultipleOutputs(training::Input &trainNetInput,
                                      const Collection<size_t> lastLayerIndices,
                                      const TensorPtr &groundTruth)
{
    for (size_t i = 0; i < lastLayerIndices.size(); i++)
    {
        size_t lastLayerIndex = lastLayerIndices[i];
        trainNetInput.add(training::groundTruthCollection, lastLayerIndex, groundTruth);
    }
}

#endif
