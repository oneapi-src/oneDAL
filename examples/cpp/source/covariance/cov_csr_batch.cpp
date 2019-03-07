/* file: cov_csr_batch.cpp */
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
!    C++ example of variance-covariance matrix computation in the batch
!    processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COVARIANCE_CSR_BATCH"></a>
 * \example cov_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters
   Input matrix is stored in the compressed sparse row format with one-based indexing
 */
const string datasetFileName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileName));

    /* Create an algorithm to compute variance-covariance matrix using the default method */
    covariance::Batch<float, covariance::fastCSR> algorithm;
    algorithm.input.set(covariance::data, dataTable);

    /* Compute a variance-covariance matrix */
    algorithm.compute();

    /* Get the computed variance-covariance matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::covariance), "Covariance matrix (upper left square 10*10) :", 10, 10);
    printNumericTable(res->get(covariance::mean),       "Mean vector:", 1, 10);

    return 0;
}
