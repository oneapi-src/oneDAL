/* file: ridge_regression_train_dense_normeq_helper_impl.i */
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
//  Implementation of auxiliary functions for ridge regression Normal Equations (normEqDense) method.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_HELPER_IMPL_I__

#include "ridge_regression_train_kernel.h"

namespace daal
{
namespace algorithms
{
namespace ridge_regression
{
namespace training
{
namespace internal
{
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status KernelHelper<algorithmFPType, cpu>::computeBetasImpl(DAAL_INT p, const algorithmFPType *a,
                                                            algorithmFPType *aCopy, DAAL_INT ny,
                                                            algorithmFPType *b, bool interceptFlag) const
{
    size_t nRidge = _ridge.getNumberOfRows();
    ReadRows<algorithmFPType, cpu> ridgeBlock(const_cast<NumericTable &>(_ridge), 0, nRidge);
    const algorithmFPType *ridge = ridgeBlock.get();

    const DAAL_INT pToFix = (interceptFlag ? p - 1 : p);

    Status st;
    if (nRidge == 1)
    {
        for(DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
        {
            aCopy[idx] += *ridge;
        }

        st |= FinalizeKernel<algorithmFPType, cpu>::solveSystem(p, aCopy, ny, b,
            ErrorRidgeRegressionInternal);
        DAAL_CHECK_STATUS_VAR(st);
    }
    else
    {
        algorithmFPType * bPtr = b;
        const size_t aSizeInBytes = p * p * sizeof(algorithmFPType);
        for (DAAL_INT j = 0; j < ny; j++, bPtr += (pToFix + 1))
        {
            daal::services::daal_memcpy_s(aCopy, aSizeInBytes, a, aSizeInBytes);
            for(DAAL_INT i = 0, idx = 0; i < pToFix; i++, idx += (p + 1))
            {
                aCopy[idx] += ridge[j];
            }

            DAAL_INT one(1);

            st |= FinalizeKernel<algorithmFPType, cpu>::solveSystem(p, aCopy, one, b,
                ErrorRidgeRegressionInternal);
            DAAL_CHECK_STATUS_VAR(st);
        }
    }
    return st;
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
