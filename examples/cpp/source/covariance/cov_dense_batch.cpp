/* file: cov_dense_batch.cpp */
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
!    C++ example of dense variance-covariance matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_DENSE_BATCH"></a>
 * \example cov_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/covcormoments_dense.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute a dense variance-covariance matrix using the default method */
    covariance::Batch<> algorithm;
    algorithm.input.set(covariance::data, dataSource.getNumericTable());

    /* Compute a dense variance-covariance matrix */
    algorithm.compute();

    /* Get the computed dense variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
    printNumericTable(res->get(covariance::mean),       "Mean vector:");

    return 0;
}
