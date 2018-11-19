/* file: pca_svd_dense_online.cpp */
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
!    C++ example of principal component analysis (PCA) using the singular value
!    decomposition (SVD) method in the online processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_SVD_DENSE_ONLINE"></a>
 * \example pca_svd_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const size_t nVectorsInBlock = 250;
const string dataFileName = "../data/online/pca_normalized.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &dataFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm for principal component analysis using the SVD method */
    pca::Online<float, pca::svdDense> algorithm;

    while(dataSource.loadDataBlock(nVectorsInBlock) == nVectorsInBlock)
    {
        /* Set the input data to the algorithm */
        algorithm.input.set(pca::data, dataSource.getNumericTable());

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
