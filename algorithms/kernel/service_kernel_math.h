/* file: service_kernel_math.h */
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
//++
//  Implementation of math functions.
//--
*/

#ifndef __SERVICE_KERNEL_MATH_H__
#define __SERVICE_KERNEL_MATH_H__

#include "service_math.h"

namespace daal
{
namespace algorithms
{
namespace internal
{

template<typename FPType, CpuType cpu>
FPType distancePow2(const FPType *a, const FPType *b, size_t dim)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += (b[i] - a[i]) * (b[i] - a[i]);
    }

    return sum;
}

template<typename FPType, CpuType cpu>
FPType distancePow(const FPType *a, const FPType *b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return sum;
}

template<typename FPType, CpuType cpu>
FPType distance(const FPType *a, const FPType *b, size_t dim, FPType p)
{
    FPType sum = 0.0;

    for (size_t i = 0; i < dim; i++)
    {
        sum += daal::internal::Math<FPType, cpu>::sPowx(b[i] - a[i], p);
    }

    return daal::internal::Math<FPType, cpu>::sPowx(sum, (FPType)1.0 / p);
}

} // namespace internal
} // namespace algorithms
} // namespace daal

#endif
