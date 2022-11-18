/* file: sgd_moment_opt_res_dense_batch.cpp */
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
!    C++ example of the Stochastic gradient descent algorithm
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SGD_MOMENT_OPT_RES_DENSE_BATCH"></a>
 * \example sgd_moment_opt_res_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

std::string datasetFileName = "../data/batch/mse.csv";

const size_t nIterations = 400;
const size_t nFeatures = 3;
const float learningRate = 0.5;
const size_t batchSize = 4;
const double accuracyThreshold = 0.0000001;

float initialPoint[nFeatures + 1] = { 8, 2, 1, 4 };

int main(int argc, char* argv[]) {
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(
        new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());
    size_t nVectors = data->getNumberOfRows();

    services::SharedPtr<optimization_solver::mse::Batch<> > mseObjectiveFunction(
        new optimization_solver::mse::Batch<>(nVectors));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables,
                                    dependentVariables);

    /* Create objects to compute the Stochastic momentum gradient descent result using the default method */
    optimization_solver::sgd::Batch<float, optimization_solver::sgd::momentum> sgdMomentumAlgorithm(
        mseObjectiveFunction);

    /* Set input objects for the the Stochastic momentum gradient descent algorithm */
    sgdMomentumAlgorithm.input.set(
        optimization_solver::iterative_solver::inputArgument,
        NumericTablePtr(new HomogenNumericTable<>(initialPoint, 1, nFeatures + 1)));
    sgdMomentumAlgorithm.parameter.learningRateSequence =
        NumericTablePtr(new HomogenNumericTable<>(1, 1, NumericTable::doAllocate, learningRate));
    sgdMomentumAlgorithm.parameter.nIterations = nIterations / 2;
    sgdMomentumAlgorithm.parameter.accuracyThreshold = accuracyThreshold;
    sgdMomentumAlgorithm.parameter.batchSize = batchSize;
    sgdMomentumAlgorithm.parameter.optionalResultRequired = true;

    /* Compute the Stochastic momentum gradient descent result */
    sgdMomentumAlgorithm.compute();

    /* Print computed the Stochastic momentum gradient descent result */
    printNumericTable(
        sgdMomentumAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
        "Minimum after first compute():");
    printNumericTable(
        sgdMomentumAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
        "Number of iterations performed:");

    /* Set optional result as an optional input */
    sgdMomentumAlgorithm.input.set(
        optimization_solver::iterative_solver::inputArgument,
        sgdMomentumAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum));
    sgdMomentumAlgorithm.input.set(optimization_solver::iterative_solver::optionalArgument,
                                   sgdMomentumAlgorithm.getResult()->get(
                                       optimization_solver::iterative_solver::optionalResult));

    /* Compute the Stochastic momentum gradient descent result */
    sgdMomentumAlgorithm.compute();

    /* Print computed the Stochastic momentum gradient descent result */
    printNumericTable(
        sgdMomentumAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum),
        "Minimum after second compute():");
    printNumericTable(
        sgdMomentumAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations),
        "Number of iterations performed:");
    return 0;
}
