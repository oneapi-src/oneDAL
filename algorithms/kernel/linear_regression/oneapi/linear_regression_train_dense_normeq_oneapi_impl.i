/* file: linear_regression_train_dense_normeq_oneapi_impl.i */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
//  Implementation of auxiliary functions for linear regression
//  Normal Equations (normEqDense) method.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_ONEAPI_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_ONEAPI_IMPL_I__

#include "linear_regression_train_kernel_oneapi.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace training
{
namespace internal
{
template <typename algorithmFPType>
services::Status BatchKernelOneAPI<algorithmFPType, training::normEqDense>::compute(NumericTable & x, NumericTable & y, NumericTable & xtx,
                                                                                    NumericTable & xty, NumericTable & beta, bool interceptFlag) const
{
    services::Status status = UpdateKernelType::compute(x, y, xtx, xty, interceptFlag);
    if (status) status = FinalizeKernelType::compute(xtx, xty, xtx, xty, beta, interceptFlag, KernelHelperOneAPI<algorithmFPType>());
    return status;
}

template <typename algorithmFPType>
services::Status OnlineKernelOneAPI<algorithmFPType, training::normEqDense>::compute(NumericTable & x, NumericTable & y, NumericTable & xtx,
                                                                          NumericTable & xty, bool interceptFlag) const
{
    return UpdateKernelType::compute(x, y, xtx, xty, interceptFlag);
}

template <typename algorithmFPType>
services::Status OnlineKernelOneAPI<algorithmFPType, training::normEqDense>::finalizeCompute(NumericTable & xtx, NumericTable & xty,
                                                                                  NumericTable & xtxFinal, NumericTable & xtyFinal,
                                                                                  NumericTable & beta, bool interceptFlag) const
{
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag, KernelHelperOneAPI<algorithmFPType>());
}

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
