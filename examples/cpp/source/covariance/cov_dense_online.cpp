/* file: cov_dense_online.cpp */
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
!    C++ example of dense variance-covariance matrix computation in the online
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_DENSE_ONLINE"></a>
 * \example cov_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/covcormoments_dense.csv";
const size_t nObservations   = 50;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute a dense variance-covariance matrix in the online processing mode using the default method */
    covariance::Online<> algorithm;
    while (dataSource.loadDataBlock(nObservations) == nObservations)
    {
        /* Set input objects for the algorithm */
        algorithm.input.set(covariance::data, dataSource.getNumericTable());

        /* Compute partial estimates */
        algorithm.compute();
    }

    /* Finalize the result in the online processing mode */
    algorithm.finalizeCompute();

    /* Get the computed dense variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance), "Covariance matrix:");
    printNumericTable(res->get(covariance::mean),       "Mean vector:");

    return 0;
}
