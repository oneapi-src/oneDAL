/* file: cor_csr_distr.cpp */
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
!    C++ example of correlation matrix computation in the distributed
!    processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CORRELATION_CSR_DISTRIBUTED">
 * \example cor_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float  algorithmFPType;     /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks         = 4;

const string datasetFileNames[] =
{
    "../data/distributed/covcormoments_csr_1.csv",
    "../data/distributed/covcormoments_csr_2.csv",
    "../data/distributed/covcormoments_csr_3.csv",
    "../data/distributed/covcormoments_csr_4.csv"
};

covariance::PartialResultPtr partialResult[nBlocks];
covariance::ResultPtr result;

void computestep1Local(size_t i);
void computeOnMasterNode();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    for(size_t i = 0; i < nBlocks; i++)
    {
        computestep1Local(i);
    }

    computeOnMasterNode();

    printNumericTable(result->get(covariance::correlation), "Correlation matrix (upper left square 10*10) :", 10, 10);
    printNumericTable(result->get(covariance::mean),        "Mean vector:", 1, 10);

    return 0;
}

void computestep1Local(size_t block)
{
    CSRNumericTable *dataTable = createSparseTable<float>(datasetFileNames[block]);

    /* Create an algorithm to compute a correlation matrix in the distributed processing mode using the default method */
    covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(covariance::data, CSRNumericTablePtr(dataTable));

    /* Compute partial estimates on local nodes */
    algorithm.compute();

    /* Get the computed partial estimates */
    partialResult[block] = algorithm.getPartialResult();
}

void computeOnMasterNode()
{
    /* Create an algorithm to compute a correlation matrix in the distributed processing mode using the default method */
    covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    for (size_t i = 0; i < nBlocks; i++)
    {
        algorithm.input.add(covariance::partialResults, partialResult[i]);
    }

    /* Set the parameter to choose the type of the output matrix */
    algorithm.parameter.outputMatrixType = covariance::correlationMatrix;

    /* Compute a partial estimate on the master node from the partial estimates on local nodes */
    algorithm.compute();

    /* Finalize the result in the distributed processing mode */
    algorithm.finalizeCompute();

    /* Get the computed correlation matrix */
    result = algorithm.getResult();
}
