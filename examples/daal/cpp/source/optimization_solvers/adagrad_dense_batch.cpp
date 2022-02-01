/* file: adagrad_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
!    C++ example of the Adaptive gradient descent algorithm
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-ADAGRAD_BATCH"></a>
 * \example adagrad_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string datasetFileName = "../data/batch/mse.csv";

const size_t nFeatures         = 3;
const double accuracyThreshold = 0.0000001;
const size_t nIterations       = 1000;
const size_t batchSize         = 1;
const float learningRate       = 1;

float startPoint[nFeatures + 1] = { 8, 2, 1, 4 };

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());
    size_t nVectors = data->getNumberOfRows();

    services::SharedPtr<optimization_solver::mse::Batch<> > mseObjectiveFunction(new optimization_solver::mse::Batch<>(nVectors));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables, dependentVariables);

    /* Create objects to compute the Adaptive gradient descent result using the default method */
    optimization_solver::adagrad::Batch<> adagradAlgorithm(mseObjectiveFunction);

    /* Set input objects for the the Adaptive gradient descent algorithm */
    adagradAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                               NumericTablePtr(new HomogenNumericTable<>(startPoint, 1, nFeatures + 1)));
    adagradAlgorithm.parameter.learningRate      = NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, learningRate));
    adagradAlgorithm.parameter.nIterations       = nIterations;
    adagradAlgorithm.parameter.accuracyThreshold = accuracyThreshold;
    adagradAlgorithm.parameter.batchSize         = batchSize;

    /* Compute the Adaptive gradient descent result */
    adagradAlgorithm.compute();

    /* Print computed the Adaptive gradient descent result */
    printNumericTable(adagradAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum), "Minimum:");
    printNumericTable(adagradAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "Number of iterations performed:");

    return 0;
}
