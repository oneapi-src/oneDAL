/* file: impl_als_csr_distr.cpp */
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
!  Content:
!    C++ example of the implicit alternating least squares (ALS) algorithm in
!    the distributed processing mode.
!
!    The program trains the implicit ALS model on a training data set.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-IMPLICIT_ALS_CSR_DISTRIBUTED"></a>
 * \example impl_als_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters */
const size_t nBlocks = 4;

/* Number of observations in training data set blocks */
const string trainDatasetFileNames[nBlocks] =
{
    "../data/distributed/implicit_als_csr_1.csv",
    "../data/distributed/implicit_als_csr_2.csv",
    "../data/distributed/implicit_als_csr_3.csv",
    "../data/distributed/implicit_als_csr_4.csv"
};

/* Number of observations in transposed training data set blocks */
const string transposedTrainDatasetFileNames[nBlocks] =
{
    "../data/distributed/implicit_als_trans_csr_1.csv",
    "../data/distributed/implicit_als_trans_csr_2.csv",
    "../data/distributed/implicit_als_trans_csr_3.csv",
    "../data/distributed/implicit_als_trans_csr_4.csv"
};

static size_t usersPartition[nBlocks + 1];
static size_t itemsPartition[nBlocks + 1];

typedef double  algorithmFPType;    /* Algorithm floating-point type */
typedef double  dataFPType;         /* Input data floating-point type */

/* Algorithm parameters */
const size_t nUsers = 46;           /* Full number of users */
const size_t nFactors = 2;          /* Number of factors */
const size_t maxIterations = 5;     /* Number of iterations in the implicit ALS training algorithm */

services::SharedPtr<CSRNumericTable> dataTable[nBlocks];
services::SharedPtr<CSRNumericTable> transposedDataTable[nBlocks];

NumericTablePtr predictedRatings[nBlocks][nBlocks];

KeyValueDataCollectionPtr userFactorsToNodes[nBlocks];
KeyValueDataCollectionPtr itemFactorsToNodes[nBlocks];

services::SharedPtr<training::DistributedPartialResultStep4> itemsPartialResultLocal[nBlocks];
services::SharedPtr<training::DistributedPartialResultStep4> usersPartialResultLocal[nBlocks];

services::SharedPtr<training::DistributedPartialResultStep1> step1LocalResult[nBlocks];
NumericTablePtr step2MasterResult;
KeyValueDataCollectionPtr step3LocalResult[nBlocks];
KeyValueDataCollectionPtr step4LocalInput[nBlocks];

void initializeModel(size_t block);
void readData(size_t block);
void trainModel();
void testModel();
void printResults();

void computeStep1Local(size_t block, services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal);
void computeStep2Master();
void computeStep3Local(size_t block, size_t *offsets,
                       services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal,
                       KeyValueDataCollectionPtr *factorsToNodes);
void computeStep4Local(size_t block, services::SharedPtr<CSRNumericTable> *dataTable,
                       services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal);

int main(int argc, char *argv[])
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        readData(i);
        initializeModel(i);

        step4LocalInput[i] = KeyValueDataCollectionPtr(new KeyValueDataCollection());
    }

    computePartialModelBlocksToNode(nBlocks, dataTable, transposedDataTable, usersPartition, itemsPartition,
                userFactorsToNodes, itemFactorsToNodes);

    trainModel();

    testModel();

    printResults();

    return 0;
}

void readData(size_t block)
{
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    dataTable[block] = services::SharedPtr<CSRNumericTable>(
        createSparseTable<dataFPType>(trainDatasetFileNames[block]));
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    transposedDataTable[block] = services::SharedPtr<CSRNumericTable>(
        createSparseTable<dataFPType>(transposedTrainDatasetFileNames[block]));
}

void initializeModel(size_t block)
{
    /* Create an algorithm object to initialize the implicit ALS model with the fastCSR method */
    training::init::Distributed<step1Local, algorithmFPType, training::init::fastCSR> initAlgorithm;
    initAlgorithm.parameter.fullNUsers = nUsers;
    initAlgorithm.parameter.nFactors = nFactors;
    initAlgorithm.parameter.seed += block;
    /* Pass a training data set to the algorithm */
    initAlgorithm.input.set(training::init::data, transposedDataTable[block]);

    /* Initialize the implicit ALS model */
    initAlgorithm.compute();

    services::SharedPtr<PartialModel> partialModelLocal = initAlgorithm.getPartialResult()->get(training::init::partialModel);

    itemsPartialResultLocal[block] = services::SharedPtr<training::DistributedPartialResultStep4>(new training::DistributedPartialResultStep4());
    itemsPartialResultLocal[block]->set(training::outputOfStep4ForStep1, partialModelLocal);
}

void computeStep1Local(size_t block, services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal)
{
    /* Create an algorithm object to perform first step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step1Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel, partialResultLocal[block]->get(training::outputOfStep4ForStep1));

    /* Compute partial results of the first step on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    step1LocalResult[block] = algorithm.getPartialResult();
}

void computeStep2Master()
{
    /* Create an algorithm object to perform second step of the implicit ALS training algorithm */
    training::Distributed<step2Master> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set the partial results of the first local step of distributed computations
       as input for the master-node algorithm */
    for (size_t i = 0; i < nBlocks; i++)
    {
        algorithm.input.add(training::inputOfStep2FromStep1, step1LocalResult[i]);
    }

    /* Compute a partial result on the master node from the partial results on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    step2MasterResult = algorithm.getPartialResult()->get(training::outputOfStep2ForStep4);
}

void computeStep3Local(size_t block, size_t *offsets, services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal,
                       KeyValueDataCollectionPtr *factorsToNodes)
{
    /* Create an algorithm object to perform third step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step3Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    services::SharedPtr<HomogenNumericTable<size_t> > offsetTable(
            new HomogenNumericTable<size_t>(&offsets[block], 1, 1));

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModel,             partialResultLocal[block]->get(training::outputOfStep4ForStep3));
    algorithm.input.set(training::partialModelBlocksToNode, factorsToNodes[block]);
    algorithm.input.set(training::offset,                   offsetTable);

    /* Compute partial results of the third step on local nodes */
    algorithm.compute();

    /* Get the computed partial results */
    step3LocalResult[block] = algorithm.getPartialResult()->get(training::outputOfStep3ForStep4);

    /* Prepare input objects for the fourth step of the distributed algorithm */
    for (size_t i = 0; i < nBlocks; i++)
    {
        (*step4LocalInput[i])[block] = (*step3LocalResult[block])[i];
    }
}

void computeStep4Local(size_t block, services::SharedPtr<CSRNumericTable> *dataTable,
                       services::SharedPtr<training::DistributedPartialResultStep4> *partialResultLocal)
{
    /* Create an algorithm object to perform fourth step of the implicit ALS training algorithm on local-node data */
    training::Distributed<step4Local> algorithm;
    algorithm.parameter.nFactors = nFactors;

    /* Set input objects for the algorithm */
    algorithm.input.set(training::partialModels,         step4LocalInput[block]);
    algorithm.input.set(training::partialData,           dataTable[block]);
    algorithm.input.set(training::inputOfStep4FromStep2, step2MasterResult);

    /* Build the implicit ALS partial model on the local node */
    algorithm.compute();

    /* Get the local implicit ALS partial models */
    partialResultLocal[block] = algorithm.getPartialResult();
}

void trainModel()
{
    for (size_t iteration = 0; iteration < maxIterations; iteration++)
    {
        /* Update partial users factors */
        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep1Local(i, itemsPartialResultLocal);
        }
        computeStep2Master();

        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep3Local(i, itemsPartition, itemsPartialResultLocal, itemFactorsToNodes);
        }

        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep4Local(i, dataTable, usersPartialResultLocal);
        }

        /* Update partial items factors */
        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep1Local(i, usersPartialResultLocal);
        }
        computeStep2Master();

        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep3Local(i, usersPartition, usersPartialResultLocal, userFactorsToNodes);
        }

        for (size_t i = 0; i < nBlocks; i++)
        {
            computeStep4Local(i, transposedDataTable, itemsPartialResultLocal);
        }
    }
}

void testModel()
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        for (size_t j = 0; j < nBlocks; j++)
        {
            /* Create an algorithm object to predict ratings based in the implicit ALS partial models */
            prediction::ratings::Distributed<step1Local> algorithm;
            algorithm.parameter.nFactors = nFactors;

            /* Set input objects for the algorithm */
            algorithm.input.set(prediction::ratings::usersPartialModel, usersPartialResultLocal[i]->get(training::outputOfStep4));
            algorithm.input.set(prediction::ratings::itemsPartialModel, itemsPartialResultLocal[j]->get(training::outputOfStep4));

            /* Predict ratings */
            algorithm.compute();

            /* Retrieve the algorithm results */
            predictedRatings[i][j] = algorithm.getResult()->get(prediction::ratings::prediction);
        }
    }
}

void printResults()
{
    for (size_t i = 0; i < nBlocks; i++)
    {
        for (size_t j = 0; j < nBlocks; j++)
        {
            cout << "Ratings for users block " << i << ", items block " << j << " :" << endl;
            printALSRatings(usersPartition[i], itemsPartition[j], predictedRatings[i][j]);
        }
    }
}
