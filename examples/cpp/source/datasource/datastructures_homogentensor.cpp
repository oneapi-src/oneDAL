/* file: datastructures_homogentensor.cpp */
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
!    C++ example of using homogeneous tensor data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_HOMOGENTENSOR">
 * \example datastructures_homogentensor.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace data_management;

int main()
{
    float data[3][3][3] = {{{1,2,3},{4,5,6},{7,8,9}},{{11,12,13},{14,15,16},{17,18,19}},{{21,22,23},{24,25,26},{27,28,29}}};

    size_t nDim = 3, dims[] = {3,3,3};

    printf("Initial data:\n");
    for(size_t i= 0;i<dims[0]*dims[1]*dims[2];i++)
    {
        printf("% 5.1f ", ((float*)data)[i]);
    }
    printf("\n");

    services::SharedPtr<HomogenTensor<> > hc = HomogenTensor<>::create(nDim, dims, (float*)data);
    checkPtr(hc.get());

    SubtensorDescriptor<float> subtensor;
    size_t fDimN = 2, fDims[] = {0,1};
    hc->getSubtensor(fDimN, fDims, 1, 2, readWrite, subtensor);

    size_t d = subtensor.getNumberOfDims();
    printf("Subtensor dimensions: %i\n", (int)(d));
    size_t n = subtensor.getSize();
    printf("Subtensor size:       %i\n", (int)(n));
    float* p = subtensor.getPtr();
    printf("Subtensor data:\n");
    for(size_t i= 0;i<n;i++)
    {
        printf("% 5.1lf ", p[i]);
    }
    printf("\n");

    p[0]=-1;

    hc->releaseSubtensor(subtensor);

    printf("Data after modification:\n");
    for(size_t i= 0;i<dims[0]*dims[1]*dims[2];i++)
    {
        printf("% 5.1f ", ((float*)data)[i]);
    }
    printf("\n");

    return 0;
}
