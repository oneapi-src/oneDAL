/* file: datastructures_rowmerged.cpp */
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
!    Row merged data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_ROWMERGED">
 * \example datastructures_rowmerged.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Row merged numeric table example" << std::endl << std::endl;

    const size_t nObservations1 = 5;
    const size_t nObservations2 = 6;
    const size_t nFeatures = 5;
    const size_t firstReadRow = 3;
    const size_t nRead = 6;
    const size_t featureIdx = 2;

    /*Example of using homogeneous numeric table*/
    double data1[nFeatures * nObservations1] =
    {
        0.0, 0.1, 0.2, 0.3, 0.4,
        1.0, 1.1, 1.2, 1.3, 1.4,
        2.0, 2.1, 2.2, 2.3, 2.4,
        3.0, 3.1, 3.2, 3.3, 3.4,
        4.0, 4.1, 4.2, 4.3, 4.4,
    };
    double data2[nFeatures * nObservations2] =
    {
        0.5, 0.6, 0.7, 0.8, 0.9,
        1.5, 1.6, 1.7, 1.8, 1.9,
        2.5, 2.6, 2.7, 2.8, 2.9,
        3.5, 3.6, 3.7, 3.8, 3.9,
        4.5, 4.6, 4.7, 4.8, 4.9,
        5.5, 5.6, 5.7, 5.8, 5.9,
    };

    /* Create row merged numeric table consisting of two homogen numeric tables */

    NumericTablePtr table1 (new HomogenNumericTable<double>(DictionaryIface::equal, data1, nFeatures, nObservations1));
    NumericTablePtr table2 (new HomogenNumericTable<double>(DictionaryIface::equal, data2, nFeatures, nObservations2));

    RowMergedNumericTable dataTable;
    dataTable.addNumericTable(table1);
    dataTable.addNumericTable(table2);

    BlockDescriptor<double> block;

    /* Read one row from merged numeric table */
    dataTable.getBlockOfRows(0, nObservations1 + nObservations2, readWrite, block);
    printArray<double>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(), "Print rows from row merged numeric table as double:");

    /* Modify row of the merged numeric table */
    double *row = block.getBlockPtr();
    for (size_t i = 0; i < nObservations1 + nObservations2; i++) row[i * nFeatures + featureIdx] *= row[i * nFeatures + featureIdx];
    dataTable.releaseBlockOfRows(block);

    dataTable.getBlockOfRows(0, nObservations1 + nObservations2, readOnly, block);
    printArray<double>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(), "Print rows from row merged numeric table as double:");
    dataTable.releaseBlockOfRows(block);

    NumericTablePtr finalizedTable = data_management::convertToHomogen<double>(dataTable);

    printNumericTable(finalizedTable.get(), "Row merged table converted to homogen numeric table");

    return 0;
}
