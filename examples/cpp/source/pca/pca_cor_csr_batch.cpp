/* file: pca_cor_csr_batch.cpp */
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
!    C++ example of principal component analysis (PCA) using the correlation
!    method in the batch processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_BATCH"></a>
 * \example pca_cor_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string dataFileName = "../data/batch/covcormoments_csr.csv";

typedef float algorithmFPType;    /* Algorithm floating-point type */

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &dataFileName);

    /* Read data from a file and create a numeric table to store input data */
    CSRNumericTablePtr dataTable(createSparseTable<float>(dataFileName));

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Batch<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance = services::SharedPtr<covariance::Batch<algorithmFPType, covariance::fastCSR> >
                                     (new covariance::Batch<algorithmFPType, covariance::fastCSR>());

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, dataTable);

    algorithm.parameter.resultsToCompute = pca::mean | pca::variance | pca::eigenvalue;
    algorithm.parameter.isDeterministic = true;
    /* Compute results of the PCA algorithm */
    algorithm.compute();

    /* Print the results */
    pca::ResultPtr result = algorithm.getResult();
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");
    printNumericTable(result->get(pca::means), "Means:");
    printNumericTable(result->get(pca::variances), "Variances:");

    return 0;
}
