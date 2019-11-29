/* file: ridge_regression_train_dense_normeq_distr_step2_impl.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of normel equations method for ridge regression
//  coefficients calculation
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__
#define __RIDGE_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__

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
Status DistributedKernel<algorithmFPType, training::normEqDense, cpu>::compute(size_t n, NumericTable ** partialxtx, NumericTable ** partialxty,
                                                                               NumericTable & xtx, NumericTable & xty) const
{
    return MergeKernelType::compute(n, partialxtx, partialxty, xtx, xty);
}

template <typename algorithmFPType, CpuType cpu>
Status DistributedKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(const NumericTable & xtx, const NumericTable & xty,
                                                                                       NumericTable & xtxFinal, NumericTable & xtyFinal,
                                                                                       NumericTable & beta, bool interceptFlag,
                                                                                       const NumericTable & ridge) const
{
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag, KernelHelper<algorithmFPType, cpu>(ridge));
}

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
