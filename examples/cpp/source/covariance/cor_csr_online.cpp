/* file: cor_csr_online.cpp */
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
!    C++ example of correlation matrix computation in the online
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_CSR_ONLINE"></a>
 * \example cor_csr_online.cpp
 */


#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType;      /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks          = 4;

const string datasetFileNames[] =
{
    "../data/online/covcormoments_csr_1.csv",
    "../data/online/covcormoments_csr_2.csv",
    "../data/online/covcormoments_csr_3.csv",
    "../data/online/covcormoments_csr_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    /* Create an algorithm to compute a correlation matrix in the online processing mode using the default method */
    covariance::Online<algorithmFPType, covariance::fastCSR> algorithm;

    /* Set the parameter to choose the type of the output matrix */
    algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    for(size_t i = 0; i < nBlocks; i++)
    {
        CSRNumericTable *dataTable = createSparseTable<float>(datasetFileNames[i]);

        /* Set input objects for the algorithm */
        algorithm.input.set(covariance::data, CSRNumericTablePtr(dataTable));

        /* Compute partial estimates */
        algorithm.compute();
    }

    /* Finalize the result in the online processing mode */
    algorithm.finalizeCompute();

    /* Get the computed correlation matrix */
    covariance::ResultPtr res = algorithm.getResult();

    printNumericTable(res->get(covariance::correlation), "Correlation matrix (upper left square 10*10) :", 10, 10);
    printNumericTable(res->get(covariance::mean),        "Mean vector:", 1, 10);

    return 0;
}
