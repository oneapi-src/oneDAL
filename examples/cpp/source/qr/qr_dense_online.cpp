/* file: qr_dense_online.cpp */
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
!    C++ example of computing QR decomposition in the online processing mode
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-QR_ONLINE"></a>
 * \example qr_dense_online.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string datasetFileName = "../data/online/qr.csv";
const size_t nRowsInBlock = 4000;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Create an algorithm to compute QR decomposition in the online processing mode */
    qr::Online<> algorithm;

    while(dataSource.loadDataBlock(nRowsInBlock) == nRowsInBlock)
    {
        algorithm.input.set( qr::data, dataSource.getNumericTable() );

        /* Compute QR decomposition */
        algorithm.compute();
    }

    /* Finalize computations and retrieve the results */
    algorithm.finalizeCompute();

    qr::ResultPtr res = algorithm.getResult();

    /* Print the results */
    printNumericTable(res->get(qr::matrixQ), "Orthogonal matrix Q:", 10);
    printNumericTable(res->get(qr::matrixR), "Triangular matrix R:");

    return 0;
}
