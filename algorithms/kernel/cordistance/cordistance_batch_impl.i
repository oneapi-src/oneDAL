/* file: cordistance_batch_impl.i */
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
//++
//  Implementation of correlation distance.
//--
*/

#include "daal_defines.h"
#include "service_math.h"
#include "service_blas.h"
#include "threading.h"
#include "service_error_handling.h"
#include "service_numeric_table.h"

static const int blockSizeDefault = 128;
#include "cordistance_full_impl.i"
#include "cordistance_up_impl.i"
#include "cordistance_lp_impl.i"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace correlation_distance
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu> bool isFull(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isUpper(NumericTableIface::StorageLayout layout);
template <typename algorithmFPType, CpuType cpu> bool isLower(NumericTableIface::StorageLayout layout);
/**
 *  \brief Kernel for Correlation distances calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
services::Status DistanceKernel<algorithmFPType, method, cpu>::compute(const size_t na, const NumericTable *const *a,
        const size_t nr, NumericTable *r[],
        const daal::algorithms::Parameter *par)
{
    NumericTable *xTable = const_cast<NumericTable *>( a[0] );  /* Input data */
    NumericTable *rTable = const_cast<NumericTable *>( r[0] );  /* Result */
    const NumericTableIface::StorageLayout rLayout = r[0]->getDataLayout();

    if(isFull<algorithmFPType, cpu>(rLayout))
    {
        return corDistanceFull<algorithmFPType, cpu>( xTable, rTable );
    }
    else
    {
        if(isLower<algorithmFPType, cpu>(rLayout))
        {
            return corDistanceLowerPacked<algorithmFPType, cpu>( xTable, rTable );
        }
        else if(isUpper<algorithmFPType, cpu>(rLayout))
        {
            return corDistanceUpperPacked<algorithmFPType, cpu>( xTable, rTable );
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
    int layoutInt = (int) layout;
    if (packed_mask & layoutInt)
    {
        return false;
    }
    return true;
}

template <typename algorithmFPType, CpuType cpu>
bool isUpper(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::upperPackedSymmetricMatrix  ||
        layout == NumericTableIface::upperPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

template <typename algorithmFPType, CpuType cpu>
bool isLower(NumericTableIface::StorageLayout layout)
{
    if (layout == NumericTableIface::lowerPackedSymmetricMatrix  ||
        layout == NumericTableIface::lowerPackedTriangularMatrix)
    {
        return true;
    }
    return false;
}

} // namespace internal

} // namespace correlation_distance

} // namespace algorithms

} // namespace daal
