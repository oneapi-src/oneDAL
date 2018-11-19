/* file: datastructures_matrix.cpp */
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
!    C++ example of using matrix data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_MATRIX">
 * \example datastructures_matrix.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Matrix numeric table example" << std::endl << std::endl;

    const size_t nObservations  = 10;
    const size_t nFeatures = 11;
    const size_t firstReadRow = 0;
    const size_t nRead = 5;
    size_t readFeatureIdx;

    /*Example of using a matrix */
    float data[nFeatures * nObservations] =
    {
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1,
        1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2,
        2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3,
        3.0f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f, 3.7f, 3.8f, 3.9f, 4,
        4.0f, 4.1f, 4.2f, 4.3f, 4.4f, 4.5f, 4.6f, 4.7f, 4.8f, 4.9f, 5,
        5.0f, 5.1f, 5.2f, 5.3f, 5.4f, 5.5f, 5.6f, 5.7f, 5.8f, 5.9f, 1,
        6.0f, 6.1f, 6.2f, 6.3f, 6.4f, 6.5f, 6.6f, 6.7f, 6.8f, 6.9f, 2,
        7.0f, 7.1f, 7.2f, 7.3f, 7.4f, 7.5f, 7.6f, 7.7f, 7.8f, 7.9f, 3,
        8.0f, 8.1f, 8.2f, 8.3f, 8.4f, 8.5f, 8.6f, 8.7f, 8.8f, 8.9f, 4,
        9.0f, 9.1f, 9.2f, 9.3f, 9.4f, 9.5f, 9.6f, 9.7f, 9.8f, 9.9f, 5
    };

    services::SharedPtr<Matrix<> > dataTable = Matrix<>::create(nFeatures, nObservations, data);
    checkPtr(dataTable.get());

    BlockDescriptor<> block;

    /* Read a block of rows */
    dataTable->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl;
    printArray<float>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(), "Print 5 rows from matrix data array as float:");
    dataTable->releaseBlockOfRows(block);

    readFeatureIdx = 2;

    /* Set new values in Matrix */
    (*dataTable)[firstReadRow][readFeatureIdx] = -1;
    (*dataTable)[firstReadRow + 1][readFeatureIdx] = -2;
    (*dataTable)[firstReadRow + 2][readFeatureIdx] = -3;

    /* Read a feature(column) and print it */
    dataTable->getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, block);
    printArray<float>(block.getBlockPtr(), 1, block.getNumberOfRows(), "Print the third feature of matrix data:");
    dataTable->releaseBlockOfColumnValues(block);

    return 0;
}
