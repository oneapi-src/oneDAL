/* file: objective_function_utils.i */
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
//  Implementation of objective function utilities
//--
*/
#include "service_math.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace objective_function
{
namespace internal
{

using namespace daal::internal;

template<typename algorithmFPType, CpuType cpu>
services::Status getXY(NumericTable *dataNT, NumericTable *dependentVariablesNT, const NumericTable *indNT,
    algorithmFPType* aX, algorithmFPType* aY, size_t nRows, size_t n, size_t p)
{
    ReadRows<int, cpu> rInd(*const_cast<NumericTable*>(indNT), 0, n);
    DAAL_CHECK_BLOCK_STATUS(rInd);
    const int* ind = rInd.get();
    ReadRows<algorithmFPType, cpu> xr(*dataNT);
    ReadRows<algorithmFPType, cpu> yr(*dependentVariablesNT, 0, nRows);
    for(size_t i = 0; i < n; ++i)
    {
        xr.next(ind[i], 1);
        services::internal::tmemcpy<algorithmFPType, cpu>(aX + i*p, xr.get(), p);
        aY[i] = yr.get()[ind[i]];
    }
    return services::Status();
}

} // namespace internal

} // namespace objective_function

} // namespace optimization_solver

} // namespace algorithms

} // namespace daal
