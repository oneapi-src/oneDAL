/* file: impl_als_dense_batch.cpp */
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
!    C++ example of the implicit alternating least squares (ALS) algorithm in
!    the batch processing mode.
!
!    The program trains the implicit ALS model on a dense training data set.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-IMPLICIT_ALS_DENSE_BATCH"></a>
 * \example impl_als_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms::implicit_als;

/* Input data set parameters */
string trainDatasetFileName            = "../data/batch/implicit_als_dense.csv";

/* Algorithm parameters */
const size_t nFactors = 2;

NumericTablePtr dataTable;
ModelPtr initialModel;
training::ResultPtr trainingResult;

void initializeModel();
void trainModel();
void testModel();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &trainDatasetFileName);

    initializeModel();

    trainModel();

    testModel();

    return 0;
}

void initializeModel()
{
    /* Read trainDatasetFileName from a file and create a numeric table to store the input data */
    FileDataSource<CSVFeatureManager> dataSource(trainDatasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the input data */
    dataSource.loadDataBlock();

    dataTable = dataSource.getNumericTable();

    /* Create an algorithm object to initialize the implicit ALS model with the default method */
    training::init::Batch<> initAlgorithm;
    initAlgorithm.parameter.nFactors = nFactors;

    /* Pass a training data set and dependent values to the algorithm */
    initAlgorithm.input.set(training::init::data, dataTable);

    /* Initialize the implicit ALS model */
    initAlgorithm.compute();

    initialModel = initAlgorithm.getResult()->get(training::init::model);
}

void trainModel()
{
    /* Create an algorithm object to train the implicit ALS model with the default method */
    training::Batch<> algorithm;

    /* Pass a training data set and dependent values to the algorithm */
    algorithm.input.set(training::data, dataTable);
    algorithm.input.set(training::inputModel, initialModel);

    algorithm.parameter.nFactors = nFactors;

    /* Build the implicit ALS model */
    algorithm.compute();

    /* Retrieve the algorithm results */
    trainingResult = algorithm.getResult();
}

void testModel()
{
    /* Create an algorithm object to predict recommendations of the implicit ALS model */
    prediction::ratings::Batch<> algorithm;
    algorithm.parameter.nFactors = nFactors;

    algorithm.input.set(prediction::ratings::model, trainingResult->get(training::model));

    algorithm.compute();

    NumericTablePtr predictedRatings = algorithm.getResult()->get(prediction::ratings::prediction);

    printNumericTable(predictedRatings, "Predicted ratings:");
}
