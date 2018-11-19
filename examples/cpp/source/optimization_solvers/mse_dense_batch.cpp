/* file: mse_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
!    C++ example of the mean squared error objective function
!******************************************************************************/


/**
 * <a name="DAAL-EXAMPLE-CPP-MSE_BATCH"></a>
 * \example mse_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

string datasetFileName = "../data/batch/mse.csv";
const size_t nFeatures = 3;

float argumentValue[nFeatures + 1] = { -1, 0.1f, 0.15f, -0.5f};

int main(int argc, char *argv[])
{
    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
            DataSource::notAllocateNumericTable,
            DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for data and values for dependent variable */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(mergedData.get());

    size_t nVectors = data->getNumberOfRows();

    /* Create the MSE objective function objects to compute the MSE objective function result using the default method */
    optimization_solver::mse::Batch<> mseObjectiveFunction(nVectors);

    /* Set input objects for the MSE objective function */
    mseObjectiveFunction.input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction.input.set(optimization_solver::mse::dependentVariables, dependentVariables);
    mseObjectiveFunction.input.set(optimization_solver::mse::argument,
                                   NumericTablePtr(new HomogenNumericTable<>(argumentValue, 1, nFeatures + 1)));
    mseObjectiveFunction.parameter.resultsToCompute =
        optimization_solver::objective_function::gradient |
        optimization_solver::objective_function::value |
        optimization_solver::objective_function::hessian;

    /* Compute the MSE objective function result */
    mseObjectiveFunction.compute();

    /* Print computed the MSE objective function result */
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::valueIdx), "Value");
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::gradientIdx), "Gradient");
    printNumericTable(mseObjectiveFunction.getResult()->get(optimization_solver::objective_function::hessianIdx), "Hessian");

    return 0;
}
