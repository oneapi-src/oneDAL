/* file: pca_cor_csr_online.cpp */
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
!    C++ example of principal component analysis (PCA) using the correlation
!    method in the online processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_ONLINE"></a>
 * \example pca_cor_csr_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType;     /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks = 4;
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

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Online<> algorithm;

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    algorithm.parameter.covariance = services::SharedPtr<covariance::Online<algorithmFPType, covariance::fastCSR> >
                                     (new covariance::Online<algorithmFPType, covariance::fastCSR>());

    for(size_t i = 0; i < nBlocks; i++)
    {
        /* Read data from a file and create a numeric table to store input data */
        CSRNumericTablePtr dataTable(createSparseTable<float>(datasetFileNames[i]));

        /* Set input objects for the algorithm */
        algorithm.input.set(pca::data, CSRNumericTablePtr(dataTable));

        /* Update PCA decomposition */
        algorithm.compute();
    }

    /* Finalize computations */
    algorithm.finalizeCompute();

    /* Print the results */
    pca::ResultPtr result = algorithm.getResult();
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
