/* file: datastructures_aos.cpp */
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
!    C++ example of using an array of structures (AOS)
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_AOS">
 * @example datastructures_aos.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

struct PointType
{
    float x;
    float y;
    int categ;
    double value;
};

int main()
{
    std::cout << "Array of structures (AOS) numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures = 4;
    PointType points[nObservations] =
    {
        {0.5f, -1.3f, 1, 100.1},
        {2.5f, -3.3f, 2, 200.2},
        {4.5f, -5.3f, 2, 350.3},
        {6.5f, -7.3f, 0, 470.4},
        {8.5f, -9.3f, 1, 270.5}
    };

    /* Construct AOS numericTable for a data array with nFeatures fields and nObservations elements*/
    AOSNumericTablePtr dataTable = AOSNumericTable::create(points, nFeatures, nObservations);
    checkPtr(dataTable.get());

    /* Add data to the numeric table */
    dataTable->setFeature<float> (0, DAAL_STRUCT_MEMBER_OFFSET(PointType, x)    , data_feature_utils::DAAL_CONTINUOUS    );
    dataTable->setFeature<float> (1, DAAL_STRUCT_MEMBER_OFFSET(PointType, y)    , data_feature_utils::DAAL_CONTINUOUS    );
    dataTable->setFeature<int>   (2, DAAL_STRUCT_MEMBER_OFFSET(PointType, categ), data_feature_utils::DAAL_CATEGORICAL, 4);
    dataTable->setFeature<double>(3, DAAL_STRUCT_MEMBER_OFFSET(PointType, value), data_feature_utils::DAAL_CONTINUOUS    );

    /* Read a block of rows */
    const size_t firstReadRow = 0;

    BlockDescriptor<double> doubleBlock;
    dataTable->getBlockOfRows(firstReadRow, nObservations, readOnly, doubleBlock);
    printArray<double>(doubleBlock.getBlockPtr(), nFeatures, doubleBlock.getNumberOfRows(), "Print AOS data structures as double:");
    dataTable->releaseBlockOfRows(doubleBlock);

    /* Read a feature (column) */
    size_t readFeatureIdx = 2;

    BlockDescriptor<int> intBlock;
    dataTable->getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, intBlock);
    printArray<int>(intBlock.getBlockPtr(), 1, intBlock.getNumberOfRows(), "Print the third feature of AOS:");
    dataTable->releaseBlockOfColumnValues(intBlock);

    return 0;
}
