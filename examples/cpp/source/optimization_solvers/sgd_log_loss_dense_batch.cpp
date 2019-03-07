/* file: sgd_log_loss_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
*******************************************************************************/

/*
!  Content:
!    C++ example of the Stochastic gradient descent algorithm with logistic loss
!    objective function
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SGD_LOG_LOSS_DENSE_BATCH"></a>
 * \example sgd_log_loss_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::optimization_solver;
using namespace daal::data_management;

string datasetFileName = "../data/batch/custom.csv";

const size_t nIterations = 1000;
const size_t nFeatures = 4;
const float  learningRate = 0.01f;
const double accuracyThreshold = 0.02;

float initialPoint[nFeatures + 1] = {1, 1, 1, 1, 1};

int main(int argc, char *argv[])
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
            DataSource::notAllocateNumericTable,
            DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    daal::services::Status s;
    NumericTablePtr data = HomogenNumericTable<>::create(nFeatures, 0, NumericTable::doNotAllocate, &s);
    checkStatus(s);
    NumericTablePtr dependentVariables = HomogenNumericTable<>::create(1, 0, NumericTable::doNotAllocate, &s);
    checkStatus(s);
    NumericTablePtr mergedData = MergedNumericTable::create(data, dependentVariables, &s);
    checkStatus(s);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());
    size_t nVectors = data.get() ? data->getNumberOfRows() : 1;
    services::SharedPtr<logistic_loss::Batch<> > batch(new logistic_loss::Batch<>(nVectors));
    batch->input.set(logistic_loss::data, data);
    batch->input.set(logistic_loss::dependentVariables, dependentVariables);

    /* Create objects to compute the Stochastic gradient descent result using the default method */
    optimization_solver::sgd::Batch<> sgdAlgorithm(batch);

    /* Set input objects for the the Stochastic gradient descent algorithm */
    sgdAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                           HomogenNumericTable<>::create(initialPoint, 1, nFeatures + 1, &s));
    checkStatus(s);
    sgdAlgorithm.parameter.learningRateSequence =
        HomogenNumericTable<>::create(1, 1, NumericTable::doAllocate, learningRate, &s);
    checkStatus(s);
    sgdAlgorithm.parameter.nIterations = nIterations;
    sgdAlgorithm.parameter.accuracyThreshold = accuracyThreshold;

    /* Compute the Stochastic gradient descent result */
    s = sgdAlgorithm.compute();
    checkStatus(s);

    /* Print computed the Stochastic gradient descent result */
    printNumericTable(sgdAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum), "Minimum:");
    printNumericTable(sgdAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "Number of iterations performed:");

    return 0;
}
