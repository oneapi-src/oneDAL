/* file: pca_cor_dense_batch.cpp */
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
!    method in the batch processing mode
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PCA_CORRELATION_DENSE_BATCH"></a>
 * \example pca_cor_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string dataFileName = "../data/batch/pca_normalized.csv";
const size_t nVectors = 1000;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &dataFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(dataFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock(nVectors);

    /* Create an algorithm for principal component analysis using the correlation method */
    pca::Batch<> algorithm;

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, dataSource.getNumericTable());
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
