/* file: pca_cor_csr_distr.cpp */
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
!    method in the distributed processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_CSR_DISTRIBUTED"></a>
 * \example pca_cor_csr_distr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float algorithmFPType;     /* Algorithm floating-point type */

/* Input data set parameters */
const size_t nBlocks         = 4;

const string datasetFileNames[] =
{
    "../data/distributed/covcormoments_csr_1.csv",
    "../data/distributed/covcormoments_csr_2.csv",
    "../data/distributed/covcormoments_csr_3.csv",
    "../data/distributed/covcormoments_csr_4.csv"
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    /* Create an algorithm for principal component analysis using the correlation method on the master node */
    pca::Distributed<step2Master> masterAlgorithm;

    for (size_t i = 0; i < nBlocks; i++)
    {
        CSRNumericTable *dataTable = createSparseTable<float>(datasetFileNames[i]);

        /* Create an algorithm to compute a variance-covariance matrix in the distributed processing mode using the default method */
        pca::Distributed<step1Local> localAlgorithm;

        /* Create an algorithm for principal component analysis using the correlation method on the local node */
        localAlgorithm.parameter.covariance = services::SharedPtr<covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR> >
                                              (new covariance::Distributed<step1Local, algorithmFPType, covariance::fastCSR>());

        /* Set input objects for the algorithm */
        localAlgorithm.input.set(pca::data, CSRNumericTablePtr(dataTable));

        /* Compute partial estimates on local nodes */
        localAlgorithm.compute();

        /* Set local partial results as input for the master-node algorithm */
        masterAlgorithm.input.add(pca::partialResults, localAlgorithm.getPartialResult());
    }

    /* Use covariance algorithm for sparse data inside the PCA algorithm */
    masterAlgorithm.parameter.covariance = services::SharedPtr<covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR> >
                                           (new covariance::Distributed<step2Master, algorithmFPType, covariance::fastCSR>());

    /* Merge and finalize PCA decomposition on the master node */
    masterAlgorithm.compute();

    masterAlgorithm.finalizeCompute();

    /* Retrieve the algorithm results */
    pca::ResultPtr result = masterAlgorithm.getResult();

    /* Print the results */
    printNumericTable(result->get(pca::eigenvalues), "Eigenvalues:");
    printNumericTable(result->get(pca::eigenvectors), "Eigenvectors:");

    return 0;
}
