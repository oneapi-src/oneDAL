/* file: saga_dense_batch.cpp */
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
!    C++ example of the SAGA algorithm
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SAGA_BATCH"></a>
 * \example saga_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"
using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

const string datasetFileName = "../data/batch/lbfgs.csv";
const size_t nFeatures = 10;   /* Number of features in training and testing data sets */

const size_t nIterations = 1000000;
const float stepLength  = 0.00001;

const float tol = 0.00000001;
float expectedPoint[nFeatures + 1] = { 11,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10};
int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for input data and dependent variables */
    NumericTablePtr data(new HomogenNumericTable<>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from input file */
    dataSource.loadDataBlock(mergedData.get());


    services::SharedPtr<optimization_solver::mse::Batch<> > mseObjectiveFunction(
        new optimization_solver::mse::Batch<float>(data->getNumberOfRows()));
    mseObjectiveFunction->input.set(optimization_solver::mse::data, data);
    mseObjectiveFunction->input.set(optimization_solver::mse::dependentVariables, dependentVariables);

    const size_t nParameters = (nFeatures + 1);
    float argument[nParameters];
    //DAAL_DATA_TYPE Wk[nParameters];
    for(int i = 0; i < nParameters; i++)
        argument[i] = 0;

    /* Create objects to compute the SAGA result using the default method */
    daal::algorithms::optimization_solver::saga::Batch<> sagaAlgorithm(mseObjectiveFunction);

    /* Set input objects for the the SAGA algorithm */
    sagaAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                           NumericTablePtr(new HomogenNumericTable<>(argument, 1, nParameters)));
    sagaAlgorithm.parameter().nIterations = nIterations;
    sagaAlgorithm.parameter().accuracyThreshold = tol;
    sagaAlgorithm.parameter().batchSize = 1;//data->getNumberOfRows();

    /* Compute the SAGA result */
    sagaAlgorithm.compute();

    /* Print computed the SAGA result */
    NumericTablePtr munimum = sagaAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum);
    printNumericTable(munimum, "Minimum:");
    printNumericTable(sagaAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "nIterations:");

    services::SharedPtr<optimization_solver::mse::Batch<> > func_check(
        new optimization_solver::mse::Batch<>(data->getNumberOfRows()));
    func_check->input.set(optimization_solver::mse::dependentVariables, dependentVariables);
    func_check->input.set(optimization_solver::mse::data, data);

    func_check->parameter().resultsToCompute = optimization_solver::objective_function::value;

    func_check->input.set(optimization_solver::mse::argument, munimum);

    func_check->compute();
    printNumericTable(func_check->getResult()->get(optimization_solver::objective_function::valueIdx),"value DAAL:");

    return 0;
}
