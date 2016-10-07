/* file: datastructures_homogentensor.cpp */
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

    HomogenTensor<float> hc(nDim, dims, (float*)data);

    SubtensorDescriptor<double> subtensor;
    size_t fDimN = 2, fDims[] = {0,1};
    hc.getSubtensor(fDimN, fDims, 1, 2, readWrite, subtensor);

    size_t d = subtensor.getNumberOfDims();
    printf("Subtensor dimensions: %i\n", (int)(d));
    size_t n = subtensor.getSize();
    printf("Subtensor size:       %i\n", (int)(n));
    double* p = subtensor.getPtr();
    printf("Subtensor data:\n");
    for(size_t i= 0;i<n;i++)
    {
        printf("% 5.1lf ", p[i]);
    }
    printf("\n");

    p[0]=-1;

    hc.releaseSubtensor(subtensor);

    printf("Data after modification:\n");
    for(size_t i= 0;i<dims[0]*dims[1]*dims[2];i++)
    {
        printf("% 5.1f ", ((float*)data)[i]);
    }
    printf("\n");

    return 0;
}
