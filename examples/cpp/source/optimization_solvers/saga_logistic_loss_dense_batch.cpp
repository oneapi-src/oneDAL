/* file: saga_logistic_loss_dense_batch.cpp */
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
 * <a name="DAAL-EXAMPLE-CPP-saga_BATCH"></a>
 * \example saga_logistic_loss_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

const string datasetFileName = "../data/batch/XM_100.csv";
const string groundTruthFileName = "../data/batch/saga_solution_100_features.csv";
const size_t nFeatures = 100;   /* Number of features in training and testing data sets */
const size_t nIterations = 100000;

/*
    stepLength bellow will be set automaticly as sklearn does for batchSize=1 case,
    and line-search will be used for n=number of terms(deterministic case)
    also it can be set by user at any value.
*/

const float tol = 0.00000001;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName,
                                                 DataSource::notAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create Numeric Tables for input data and dependent variables */
    NumericTablePtr data(new HomogenNumericTable<float>(nFeatures, 0, NumericTable::doNotAllocate));
    NumericTablePtr dependentVariables(new HomogenNumericTable<float>(1, 0, NumericTable::doNotAllocate));
    NumericTablePtr mergedData(new MergedNumericTable(data, dependentVariables));

    /* Retrieve the data from input file */
    dataSource.loadDataBlock(mergedData.get());

    services::SharedPtr<optimization_solver::logistic_loss::Batch<float> > func(new optimization_solver::logistic_loss::Batch<float>(data->getNumberOfRows()));
    func->input.set(optimization_solver::logistic_loss::data, data);
    func->input.set(optimization_solver::logistic_loss::dependentVariables, dependentVariables);

    const size_t nParameters = (nFeatures + 1);
    float argument[nParameters];

    for(int i = 0; i < nParameters; i++)
        argument[i] = 0;
    argument[0] = 0;
    argument[1] = 0;

    func->parameter().penaltyL1 = 0.06;
    func->parameter().penaltyL2 = 0;    /* penalty L2 is set to zero value as sklearn doesn`t support l1 and l2 similtaniously in curent moment */
    func->parameter().interceptFlag = false;
    func->parameter().resultsToCompute = optimization_solver::objective_function::gradient;

    /* Create objects to compute the SAGA result using the default method */
    daal::algorithms::optimization_solver::saga::Batch<float> sagaAlgorithm(func);

    /* Set input objects for the the SAGA algorithm */
    sagaAlgorithm.input.set(optimization_solver::iterative_solver::inputArgument,
                            HomogenNumericTable<float>::create(argument, 1, nParameters));

    sagaAlgorithm.parameter().nIterations = nIterations;
    sagaAlgorithm.parameter().accuracyThreshold = tol;
    sagaAlgorithm.parameter().batchSize = 1;//data->getNumberOfRows();

    sagaAlgorithm.compute();


    /* Print computed the SAGA result */
    NumericTablePtr munimum = sagaAlgorithm.getResult()->get(optimization_solver::iterative_solver::minimum);
    printNumericTable(munimum, "Minimum:");
    printNumericTable(sagaAlgorithm.getResult()->get(optimization_solver::iterative_solver::nIterations), "nIterations:");

    ///* Check the value of objective function */
    services::SharedPtr<optimization_solver::logistic_loss::Batch<float> > func_check(new optimization_solver::logistic_loss::Batch<float>(data->getNumberOfRows()));
    func_check->input.set(optimization_solver::logistic_loss::data, data);
    func_check->input.set(optimization_solver::logistic_loss::dependentVariables, dependentVariables);


    func_check->parameter().penaltyL1 = 0.06;
    func_check->parameter().penaltyL2 = 0;//float(1.0)/float((2*100.0*4096));
    func_check->parameter().interceptFlag = false;

    func_check->parameter().resultsToCompute = optimization_solver::objective_function::value; /*contains the smooth and not smooth part */

    FileDataSource<CSVFeatureManager> groundTruthDS(groundTruthFileName,
                                                        DataSource::notAllocateNumericTable,
                                                        DataSource::doDictionaryFromContext);

    NumericTablePtr groundTruthNT(new HomogenNumericTable<float>(1, 0, NumericTable::doNotAllocate));

    /* Retrieve the data from input file */
    groundTruthDS.loadDataBlock(groundTruthNT.get());

    func_check->input.set(optimization_solver::logistic_loss::argument, groundTruthNT);
    func_check->compute();
    printNumericTable(func_check->getResult()->get(optimization_solver::objective_function::valueIdx),"groundTruth:");

    func_check->input.set(optimization_solver::logistic_loss::argument, munimum);
    func_check->compute();
    printNumericTable(func_check->getResult()->get(optimization_solver::objective_function::valueIdx),"value DAAL:");

    return 0;
}
