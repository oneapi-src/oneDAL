/* file: linear_regression_train_kernel.h */
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
//  Declaration of structure containing kernels for linear regression
//  training.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_KERNEL_H__
#define __LINEAR_REGRESSION_TRAIN_KERNEL_H__

#include "numeric_table.h"
#include "algorithm_base_common.h"
#include "linear_regression_training_types.h"
#include "linear_model_train_normeq_kernel.h"
#include "linear_model_train_qr_kernel.h"
#include "algorithm_kernel.h"

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
    Status computeBetasImpl(DAAL_INT p, const algorithmFPType *a,algorithmFPType *aCopy,
                            DAAL_INT ny, algorithmFPType *b, bool inteceptFlag) const;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::normEqDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernel <algorithmFPType, cpu>     UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(const NumericTable &x, const NumericTable &y, NumericTable &xtx,
                   NumericTable &xty, NumericTable &beta, bool interceptFlag) const;
};

template <typename algorithmFPType, CpuType cpu>
class BatchKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::UpdateKernel <algorithmFPType, cpu>     UpdateKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(const NumericTable &x, const NumericTable &y, NumericTable &r,
                   NumericTable &qty, NumericTable &beta, bool interceptFlag) const;
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
                           NumericTable &beta, bool interceptFlag) const;
};

template <typename algorithmFPType, CpuType cpu>
class OnlineKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::UpdateKernel <algorithmFPType, cpu>     UpdateKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(const NumericTable &x, const NumericTable &y, NumericTable &r, NumericTable &qty,
                   bool interceptFlag) const;
    Status finalizeCompute(const NumericTable &r, const NumericTable &qty, NumericTable &rFinal,
                           NumericTable &qtyFinal, NumericTable &beta, bool interceptFlag) const;
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
                           NumericTable &beta, bool interceptFlag) const;
};

template <typename algorithmFPType, CpuType cpu>
class DistributedKernel<algorithmFPType, training::qrDense, cpu> : public daal::algorithms::Kernel
{
    typedef linear_model::qr::training::internal::MergeKernel   <algorithmFPType, cpu>    MergeKernelType;
    typedef linear_model::qr::training::internal::FinalizeKernel<algorithmFPType, cpu>    FinalizeKernelType;
public:
    Status compute(size_t n, NumericTable **partialr, NumericTable **partialqty,
                   NumericTable &r, NumericTable &qty) const;
    Status finalizeCompute(const NumericTable &r, const NumericTable &qty, NumericTable &rFinal,
                           NumericTable &qtyFinal, NumericTable &beta, bool interceptFlag) const;
};

} // internal
} // training
} // linear_regression
} // algorithms
} // daal

#endif
