/* file: linear_regression_train_dense_normeq_distr_step2_impl.i */
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
//  Implementation of normel equations method for linear regression
//  coefficients calculation
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__
#define __LINEAR_REGRESSION_TRAIN_DENSE_NORMEQ_DISTR_STEP2_IMPL_I__

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
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, CpuType cpu>
Status DistributedKernel<algorithmFPType, training::normEqDense, cpu>::compute(
    size_t n, NumericTable **partialxtx, NumericTable **partialxty,
    NumericTable &xtx, NumericTable &xty) const
{
    return MergeKernelType::compute(n, partialxtx, partialxty, xtx, xty);
}

template <typename algorithmFPType, CpuType cpu>
Status DistributedKernel<algorithmFPType, training::normEqDense, cpu>::finalizeCompute(
    const NumericTable &xtx, const NumericTable &xty, NumericTable &xtxFinal, NumericTable &xtyFinal,
    NumericTable &beta, bool interceptFlag) const
{
    return FinalizeKernelType::compute(xtx, xty, xtxFinal, xtyFinal, beta, interceptFlag,
                                       KernelHelper<algorithmFPType, cpu>());
}

} // internal
} // training
} // linear_regression
} // algorithms
} // daal

#endif
