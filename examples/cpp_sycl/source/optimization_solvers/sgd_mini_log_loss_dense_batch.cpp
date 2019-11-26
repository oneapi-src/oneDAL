/* file: sgd_mini_log_loss_dense_batch.cpp */
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
!  Content:
!    C++ example of the Stochastic gradient descent algorithm with logistic loss
!    objective function with DPC++ interfaces
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SGD_LOG_LOSS_DENSE_BATCH"></a>
 * \example sgd_mini_log_loss_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::optimization_solver;
using namespace daal::data_management;

string datasetFileName = "../data/batch/custom.csv";

const size_t nIterations       = 1000;
const size_t nFeatures         = 4;
const float learningRate       = 0.01f;
const double accuracyThreshold = 0.02;
const size_t batchSize         = 4;

float initialPoint[nFeatures + 1] = { 1, 1, 1, 1, 1 };

int main(int argc, char * argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);
    daal::services::Status s;

    /* Initialize sycl context */
    for (const auto & deviceSelector : getListOfDevices())
    {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);
        std::cout << "Running on " << nameDevice << "\n\n";
        services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        /* Create Numeric Tables for data and values for dependent variable */
        NumericTablePtr data = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate, &s);
        checkStatus(s);
        NumericTablePtr dependentVariables = SyclHomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate, &s);
        checkStatus(s);
        NumericTablePtr mergedData = MergedNumericTable::create(data, dependentVariables, &s);
        checkStatus(s);

        /* Retrieve the data from the input file */
        dataSource.loadDataBlock(mergedData.get());

        size_t nVectors = data.get() ? data->getNumberOfRows() : 1;
        services::SharedPtr<logistic_loss::Batch<float> > logLoss(new logistic_loss::Batch<float>(nVectors));
        logLoss->input.set(logistic_loss::data, data);
        logLoss->input.set(logistic_loss::dependentVariables, dependentVariables);

        /* Create objects to compute the Stochastic gradient descent result using the default method */
        optimization_solver::sgd::Batch<float, optimization_solver::sgd::miniBatch> sgdAlgorithm(logLoss);

        /* Set input objects for the the Stochastic gradient descent algorithm */
        cl::sycl::buffer<float, 1> initialPointBuff(initialPoint, cl::sycl::range<1>(nFeatures + 1));
        sgdAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                               SyclHomogenNumericTable<>::create(initialPointBuff, 1, nFeatures + 1, &s));
        checkStatus(s);
        sgdAlgorithm.parameter.learningRateSequence = HomogenNumericTable<>::create(1, 1, NumericTable::doAllocate, learningRate, &s);
        checkStatus(s);
        sgdAlgorithm.parameter.nIterations       = nIterations;
        sgdAlgorithm.parameter.accuracyThreshold = accuracyThreshold;
        sgdAlgorithm.parameter.batchSize         = batchSize;

        /* Compute the Stochastic gradient descent result */
        s = sgdAlgorithm.compute();
        checkStatus(s);

        /* Print computed the Stochastic gradient descent result */
        printNumericTable(sgdAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum), "Minimum:");
        printNumericTable(sgdAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "Number of iterations performed:");
    }
    return 0;
}
