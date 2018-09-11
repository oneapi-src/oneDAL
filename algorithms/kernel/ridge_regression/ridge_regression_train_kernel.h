/* file: ridge_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for ridge regression training.
//--
*/

#ifndef __RIDGE_REGRESSION_TRAIN_KERNEL_H__
#define __RIDGE_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "ridge_regression_training_types.h"
#include "linear_model_train_normeq_kernel.h"

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
using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::linear_model::normal_equations::training::internal;

template <typename algorithmFPType, training::Method method, CpuType cpu>
class BatchKernel
{};

template <typename algorithmFPType, CpuType cpu>
class KernelHelper : public KernelHelperIface<algorithmFPType, cpu>
{
public:
    KernelHelper(const NumericTable &ridge) : _ridge(ridge) {}
    Status computeBetasImpl(DAAL_INT p, const algorithmFPType *a,algorithmFPType *aCopy,
                            DAAL_INT ny, algorithmFPType *b, bool inteceptFlag) const;
protected:
    const NumericTable &_ridge;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel <algorithmFPType, cpu>     UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(const NumericTable &x, const NumericTable &y, NumericTable &xtx,
                   NumericTable &xty, NumericTable &beta, bool interceptFlag,
                   const NumericTable &ridge) const;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
class OnlineKernel
{};

template <typename algorithmFPType, CpuType cpu>
class OnlineKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel <algorithmFPType, cpu>     UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(const NumericTable &x, const NumericTable &y, NumericTable &xtx, NumericTable &xty,
                   bool interceptFlag) const;

    Status finalizeCompute(const NumericTable &xtx, const NumericTable &xty, NumericTable &xtxFinal, NumericTable &xtyFinal,
                           NumericTable &beta, bool interceptFlag, const NumericTable &ridge) const;
};

template <typename algorithmFPType, training::Method method, CpuType cpu>
class DistributedKernel
{};

template <typename algorithmFPType, CpuType cpu>
class DistributedKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::MergeKernel   <algorithmFPType, cpu>    MergeKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(size_t n, NumericTable **partialxtx, NumericTable **partialxty,
                   NumericTable &xtx, NumericTable &xty) const;

    Status finalizeCompute(const NumericTable &xtx, const NumericTable &xty, NumericTable &xtxFinal, NumericTable &xtyFinal,
                           NumericTable &beta, bool interceptFlag, const NumericTable &ridge) const;
};

} // namespace internal
} // namespace training
} // namespace ridge_regression
} // namespace algorithms
} // namespace daal

#endif
