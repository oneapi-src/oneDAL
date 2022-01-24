/* file: cosdistance_batch_impl.i */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
//++
//  Implementation of cosine distance.
//--
*/

#include "services/daal_defines.h"
#include "src/externals/service_math.h"
#include "src/externals/service_blas.h"
#include "src/threading/threading.h"
#include "src/algorithms/service_error_handling.h"
#include "src/data_management/service_numeric_table.h"

static const int blockSizeDefault = 128;
#include "src/algorithms/cosdistance/cosdistance_full_impl.i"
#include "src/algorithms/cosdistance/cosdistance_up_impl.i"
#include "src/algorithms/cosdistance/cosdistance_lp_impl.i"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace cosine_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout);

/**
 *  \brief Kernel for Cosine distances calculation
 */
template <typename algorithmFPType, Method method, CpuType cpu>
services::Status DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable * const * a, const size_t nr,
                                                                       NumericTable * r[], const daal::algorithms::Parameter * par)
{
    NumericTable * xTable                          = const_cast<NumericTable *>(a[0]); /* Input data */
    NumericTable * rTable                          = const_cast<NumericTable *>(r[0]); /* Output data */
    const NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    if (isFull<algorithmFPType, cpu>(rLayout))
    {
        return cosDistanceFull<algorithmFPType, cpu>(xTable, rTable);
    }
    else
    {
        if (isLower<algorithmFPType, cpu>(rLayout))
        {
            return cosDistanceLowerPacked<algorithmFPType, cpu>(xTable, rTable);
        }
        else if (isUpper<algorithmFPType, cpu>(rLayout))
        {
            return cosDistanceUpperPacked<algorithmFPType, cpu>(xTable, rTable);
        }
        else
        {
            return services::Status(services::ErrorIncorrectTypeOfOutputNumericTable);
        }
    }
}

template <typename algorithmFPType, CpuType cpu>
bool isFull(NumericTableIface::StorageLayout layout)
{
    int layoutInt = (int)layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix || layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix || layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

} // namespace internal

} // namespace cosine_distance

} // namespace algorithms

} // namespace daal
