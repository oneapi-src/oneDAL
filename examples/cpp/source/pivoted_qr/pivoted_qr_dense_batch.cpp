/* file: pivoted_qr_dense_batch.cpp */
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
!    C++ example of computing pivoted QR decomposition
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-PIVOTED_QR_BATCH"></a>
 * \example pivoted_qr_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/batch/qr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Create an algorithm to compute pivoted QR decomposition */
    pivoted_qr::Batch<> algorithm;

    algorithm.input.set(pivoted_qr::data, dataSource.getNumericTable());

    /* Compute pivoted QR decomposition */
    algorithm.compute();

    pivoted_qr::ResultPtr res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(pivoted_qr::matrixQ), "Orthogonal matrix Q:", 10);
    printNumericTable(res->get(pivoted_qr::matrixR), "Triangular matrix R:");
    printNumericTable(res->get(pivoted_qr::permutationMatrix), "Permutation matrix P:");

    return 0;
}
