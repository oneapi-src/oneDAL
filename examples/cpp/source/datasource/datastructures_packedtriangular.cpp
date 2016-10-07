/* file: datastructures_packedtriangular.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
!  Content:
!    C++ example of using packed data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_PACKEDTRIANGULAR">
 * \example datastructures_packedtriangular.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Packed triangular matrix example" << std::endl << std::endl;

    const size_t nDim  = 5;
    const size_t firstReadRow = 0;
    const size_t nRead = 5;
    size_t readFeatureIdx;

    /*Example of using a packed triangular matrix */
    double data[nDim * (nDim + 1) / 2] =
    {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4
    };

    PackedTriangularMatrix<NumericTableIface::lowerPackedTriangularMatrix, double> dataTable(data, nDim);

    BlockDescriptor<double> block;

    /* Read a block of rows */
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl;
    printArray<double>(block.getBlockPtr(), nDim, block.getNumberOfRows(), "Print 3 rows from packed triangular matrix as double:");

    /* Read a feature(column) and write into it */
    readFeatureIdx = 2;
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nDim, readWrite, block);
    printArray<double>(block.getBlockPtr(), 1, block.getNumberOfRows(), "Print the third feature of packed triangular matrix:");

    /* Set new value to a buffer and release it */
    double* dataBlock = block.getBlockPtr();
    dataBlock[readFeatureIdx - 1] = -1;
    dataBlock[readFeatureIdx + 1] = -2;
    dataTable.releaseBlockOfColumnValues(block);

    /* Read a block of rows. Ensure that data has changed */
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl;
    printArray<double>(block.getBlockPtr(), nDim, block.getNumberOfRows(), "Print 3 rows from packed triangular matrix as double:");

    return 0;
}
