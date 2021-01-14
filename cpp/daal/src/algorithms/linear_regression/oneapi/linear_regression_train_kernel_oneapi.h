/* file: linear_regression_train_kernel_oneapi.h */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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
//  Declaration of structure containing kernels for linear regression
//  training.
//--
*/

#ifndef __LINEAR_REGRESSION_TRAIN_KERNEL_ONEAPI_H__
#define __LINEAR_REGRESSION_TRAIN_KERNEL_ONEAPI_H__

#include "data_management/data/numeric_table.h"
#include "algorithms/algorithm_base_common.h"
#include "algorithms/linear_regression/linear_regression_training_types.h"
#include "src/algorithms/linear_model/oneapi/linear_model_train_normeq_kernel_oneapi.h"
#include "algorithms/algorithm_kernel.h"

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
template <typename algorithmFPType, training::Method method>
class BatchKernelOneAPI
{};

template <typename algorithmFPType>
class KernelHelperOneAPI : public linear_model::normal_equations::training::internal::KernelHelperOneAPIIface<algorithmFPType>
{
public:
    services::Status computeBetasImpl(const size_t p, services::internal::Buffer<algorithmFPType> & a, const size_t ny,
                                      services::internal::Buffer<algorithmFPType> & b, const bool inteceptFlag) const;
    services::Status copyBetaToResult(const services::internal::Buffer<algorithmFPType> & betaTmp,
                                      services::internal::Buffer<algorithmFPType> & betaRes, const size_t nBetas, const size_t nResponses,
                                      const bool interceptFlag) const;
};

template <typename algorithmFPType>
class BatchKernelOneAPI<algorithmFPType, training::normEqDense> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernelOneAPI<algorithmFPType> UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernelOneAPI<algorithmFPType> FinalizeKernelType;

public:
    services::Status compute(NumericTable & x, NumericTable & y, NumericTable & xtx, NumericTable & xty, NumericTable & beta,
                             bool interceptFlag) const;
};

template <typename algorithmFPType, training::Method method>
class OnlineKernelOneAPI
{};

template <typename algorithmFPType>
class OnlineKernelOneAPI<algorithmFPType, training::normEqDense> : public daal::algorithms::Kernel
{
    typedef linear_model::normal_equations::training::internal::UpdateKernelOneAPI<algorithmFPType> UpdateKernelType;
    typedef linear_model::normal_equations::training::internal::FinalizeKernelOneAPI<algorithmFPType> FinalizeKernelType;

public:
    services::Status compute(NumericTable & x, NumericTable & y, NumericTable & xtx, NumericTable & xty, bool interceptFlag) const;
    services::Status finalizeCompute(NumericTable & xtx, NumericTable & xty, NumericTable & xtxFinal, NumericTable & xtyFinal, NumericTable & beta,
                                     bool interceptFlag) const;
};

} // namespace internal
} // namespace training
} // namespace linear_regression
} // namespace algorithms
} // namespace daal

#endif
