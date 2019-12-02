/* file: linear_regression_train_dense_qr_distr_step2_impl.i */
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
//  Implementation of normel equations method for linear regression
//  coefficients calculation
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_QR_DISTR_STEP2_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_QR_DISTR_STEP2_IMPL_I__

#include "linear_regression_train_kernel.h"

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
using namespace daal::algorithms::linear_model::qr::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status DistributedKernel<algorithmFPType, training::qrDense, cpu>::compute(size_t n, NumericTable ** partialr, NumericTable ** partialqty,
                                                                           NumericTable & r, NumericTable & qty) const
{
    return MergeKernelType::compute(n, partialr, partialqty, r, qty);
}

template <typename algorithmFPType, CpuType cpu>
services::Status DistributedKernel<algorithmFPType, training::qrDense, cpu>::finalizeCompute(const NumericTable & r, const NumericTable & qty,
                                                                                             NumericTable & rFinal, NumericTable & qtyFinal,
                                                                                             NumericTable & beta, bool interceptFlag) const
{
    return FinalizeKernelType::compute(r, qty, rFinal, qtyFinal, beta, interceptFlag);
}

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
