/* file: cor_dense_batch.cpp */
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
!    C++ example of dense correlation matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_DENSE_BATCH"></a>
 * \example cor_dense_batch.cpp
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

    /* Create an algorithm to compute a dense correlation matrix using the default method */
    covariance::Batch<> algorithm;
    algorithm.input.set(covariance::data, dataSource.getNumericTable());

    /* Set the parameter to choose the type of the output matrix */
    algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    /* Compute a dense correlation matrix */
    algorithm.compute();

    /* Get the computed dense correlation matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::correlation), "Correlation matrix:");
    printNumericTable(res->get(covariance::mean),        "Mean vector:");

    return 0;
}
